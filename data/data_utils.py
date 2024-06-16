"""Mesh data utilities."""
import random
import networkx as nx
import numpy as np
from six.moves import range
import trimesh
from scipy.spatial.transform import Rotation


def write_obj(vertices, faces, file_path, transpose=True, scale=1):
    """Write vertices and faces to obj."""
    if transpose:
        vertices = vertices[:, [1, 2, 0]]
    vertices *= scale
    if faces is not None:
        if min(min(faces)) == 0:
            f_add = 1
        else:
            f_add = 0
    with open(file_path, "w") as f:
        for v in vertices:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for face in faces:
            line = "f"
            for i in face:
                line += " {}".format(i + f_add)
            line += "\n"
            f.write(line)


def write_mesh(vertices, faces, file_path, post_process=False):
         
    if min(min(faces)) == 1:
        faces = (np.array(faces) - 1).tolist()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if post_process:
        mesh.merge_vertices()
        mesh.fix_normals()
    mesh.export(file_path)


def to_mesh(vertices, faces, transpose=True):
    if transpose:
        vertices = vertices[:, [1, 2, 0]]
        
    if faces.min() == 1:
        faces = (np.array(faces) - 1).tolist()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


def quantize_verts(verts, n_bits=8):
    """Convert vertices in [-1., 1.] to discrete values in [0, n_bits**2 - 1]."""
    scale = 0.5 #1.0
    min_range = -scale
    max_range = scale
    range_quantize = 2**n_bits - 1
    verts_quantize = (verts - min_range) * range_quantize / (max_range - min_range)
    return verts_quantize.astype("int32")


def dequantize_verts(verts, n_bits=8, add_noise=False):
    """Convert quantized vertices to floats."""
    scale = 0.5 #1.0
    min_range = -scale
    max_range = scale
    range_quantize = 2**n_bits - 1
    verts = verts.astype("float32")
    verts = verts * (max_range - min_range) / range_quantize + min_range
    if add_noise:
        verts += np.random.uniform(size=verts.shape) * (1 / range_quantize)
    return verts


def face_to_cycles(face):
    """Find cycles in face."""
    g = nx.Graph()
    for v in range(len(face) - 1):
        g.add_edge(face[v], face[v + 1])
    g.add_edge(face[-1], face[0])
    return list(nx.cycle_basis(g))


def flatten_faces(faces):
    """Converts from list of faces to flat face array with stopping indices."""
    if not faces:
        return np.array([0])
    else:
        l = [f + [-1] for f in faces[:-1]]
        l += [faces[-1] + [-2]]
        return (
            np.array([item for sublist in l for item in sublist]) + 2
        )  # pylint: disable=g-complex-comprehension


def unflatten_faces(flat_faces):
    """Converts from flat face sequence to a list of separate faces."""

    def group(seq):
        g = []
        for el in seq:
            if el == 0 or el == -1:
                yield g
                g = []
            else:
                g.append(el - 1)
        yield g

    outputs = list(group(flat_faces - 1))[:-1]
    # Remove empty faces
    return [o for o in outputs if len(o) > 2]


def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero."""
    vert_min = vertices.min(axis=0)
    vert_max = vertices.max(axis=0)
    vert_center = 0.5 * (vert_min + vert_max)
    # vert_center = np.mean(vertices, axis=0)
    return vertices - vert_center


def normalize_vertices_scale(vertices):
    """Scale the vertices so that the long diagonal of the bounding box is one."""
    vert_min = vertices.min(axis=0)
    vert_max = vertices.max(axis=0)
    extents = vert_max - vert_min
    scale = np.sqrt(np.sum(extents**2))
    # scale = np.max(np.abs(vertices))
    return vertices / (scale + 1e-6)


def quantize_process_mesh(vertices, faces, tris=None, quantization_bits=8):
    """Quantize vertices, remove resulting duplicates and reindex faces."""
    vertices = quantize_verts(vertices, quantization_bits)
    vertices, inv = np.unique(vertices, axis=0, return_inverse=True)

    # Sort vertices by z then y then x.
    sort_inds = np.lexsort(vertices.T)
    vertices = vertices[sort_inds]

    # Re-index faces and tris to re-ordered vertices.
    faces = [np.argsort(sort_inds)[inv[f]] for f in faces]
    if tris is not None:
        tris = np.array([np.argsort(sort_inds)[inv[t]] for t in tris])

    # Merging duplicate vertices and re-indexing the faces causes some faces to
    # contain loops (e.g [2, 3, 5, 2, 4]). Split these faces into distinct
    # sub-faces.
    sub_faces = []
    for f in faces:
        cliques = face_to_cycles(f)
        for c in cliques:
            c_length = len(c)
            # Only append faces with more than two verts.
            if c_length > 2:
                d = np.argmin(f)
                # Cyclically permute faces just that first index is the smallest.
                sub_faces.append([f[(d + i) % c_length] for i in range(c_length)])
                
                # d = np.argmin(c)
                # # Cyclically permute faces just that first index is the smallest.
                # sub_faces.append([c[(d + i) % c_length] for i in range(c_length)])
    faces = sub_faces
    if tris is not None:
        tris = np.array([v for v in tris if len(set(v)) == len(v)])

    # Sort faces by lowest vertex indices. If two faces have the same lowest
    # index then sort by next lowest and so on.
    faces.sort(key=lambda f: tuple(sorted(f)))
    if tris is not None:
        tris = tris.tolist()
        tris.sort(key=lambda f: tuple(sorted(f)))
        tris = np.array(tris)

    # After removing degenerate faces some vertices are now unreferenced.
    # Remove these.
    num_verts = vertices.shape[0]
    vert_connected = np.equal(
        np.arange(num_verts)[:, None], np.hstack(faces)[None]
    ).any(axis=-1)
    vertices = vertices[vert_connected]

    # Re-index faces and tris to re-ordered vertices.
    vert_indices = np.arange(num_verts) - np.cumsum(1 - vert_connected.astype("int"))
    faces = [vert_indices[f].tolist() for f in faces]
    if tris is not None:
        tris = np.array([vert_indices[t].tolist() for t in tris])

    return vertices, faces, tris


def process_mesh(vertices, faces, quantization_bits=8, flatten=False, augment=True, augment_dict=None):
    """Process mesh vertices and faces."""

    # # Transpose so that z-axis is vertical.
    # vertices = vertices[:, [2, 0, 1]]

    # Translate the vertices so that bounding box is centered at zero.
    vertices = center_vertices(vertices)

    if augment:
        vertices = augment_mesh(vertices, **augment_dict)

    # Scale the vertices so that the long diagonal of the bounding box is equal
    # to one.
    vertices = normalize_vertices_scale(vertices)

    # Quantize and sort vertices, remove resulting duplicates, sort and reindex
    # faces.
    vertices, faces, _ = quantize_process_mesh(
        vertices, faces, quantization_bits=quantization_bits
    )
    vertices = dequantize_verts(vertices, n_bits=quantization_bits, add_noise=False)

    # Flatten faces and add 'new face' = 1 and 'stop' = 0 tokens.
    if flatten:
        faces = flatten_faces(faces)

    # Discard degenerate meshes without faces.
    return {
        "vertices": vertices,
        "faces": faces,
    }


def load_process_mesh(mesh_obj_path, quantization_bits=8, augment=False, augment_dict=None):
    """Load obj file and process."""
    # Load mesh
    mesh = trimesh.load(mesh_obj_path, force='mesh', process=False)
    return process_mesh(mesh.vertices, mesh.faces, quantization_bits, augment=augment, augment_dict=augment_dict)


def augment_mesh(vertices, scale_min=0.95, scale_max=1.05, rotation=0., jitter_strength=0.):
    '''scale vertices by a factor in [0.75, 1.25]'''
    
    # vertices [nv, 3]
    for i in range(3):
        # Generate a random scale factor
        scale = random.uniform(scale_min, scale_max)    

        # independently applied scaling across each axis of vertices
        vertices[:, i] *= scale
    
    if rotation != 0.:
        axis = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]
        radian = np.pi / 180 * rotation
        rotation = Rotation.from_rotvec(radian * np.array(axis))
        vertices =rotation.apply(vertices)
        
        
    if jitter_strength != 0.:
        jitter_amount = np.random.uniform(-jitter_strength, jitter_strength)
        vertices += jitter_amount
    
        
    return vertices


def pad_mesh(vertices, faces, patch_size = 1):
    
    if len(faces) % patch_size != 0:
        # pad vertices
        vertices = vertices.tolist()
        vertices.append([0., 0., 0.])
        
        # pad faces
        pad_idx = len(vertices) - 1
        pad_num = patch_size - len(faces) % patch_size
        for _ in range(pad_num):
            faces.append([pad_idx, pad_idx, pad_idx])
    
    assert len(faces) % patch_size == 0, 'Faces should be padding'

    return {
        "vertices": vertices,
        "faces": faces,
    }