import numpy as np
import torch
import trimesh
import os
from metric_utils import compute_all_metrics
from tqdm import tqdm

def pc_norm(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def sample_pc(mesh_dir, eval_num=None, pc_num=1024, return_path=False):
    
    if os.path.isdir(mesh_dir):
        mesh_list = [os.path.join(mesh_dir, file) for file in os.listdir(mesh_dir)]
    else: 
        with open(mesh_dir) as f:
            mesh_list = [path.strip('\n') for path in f.readlines()]
    
    eval_num = eval_num if eval_num is not None else len(mesh_list)
    points_list = []
    path_list = []
        
    for mesh_path in tqdm(mesh_list):
        try:
            mesh = trimesh.load(mesh_path, force='mesh', process=False)
            pc = mesh.as_open3d.sample_points_uniformly(number_of_points = pc_num)
            points = pc_norm(np.asarray(pc.points))
            points_list.append(points[None, :])
            if return_path:
                path_list.append(mesh_path)
        except Exception as e:
            print(e)
        
        if len(points_list) == eval_num:
            break
    
    if return_path:
        return np.concatenate(points_list, axis=0), path_list
        
    return np.concatenate(points_list, axis=0)

def main(sample_dir, ref_dir = None, batch_size=32):
    sample_pcs = sample_pc(sample_dir)
    ref_pcs = sample_pc(ref_dir, len(sample_pcs))
    sample_pcs, ref_pcs = torch.tensor(sample_pcs), torch.tensor(ref_pcs)
    sample_pcs, ref_pcs = sample_pcs[torch.randperm(len(sample_pcs))], ref_pcs[torch.randperm(len(ref_pcs))]
    min_len = min(len(sample_pcs), len(ref_pcs), 1000)
    print('The number of evaluated samples:', min_len)
    res = compute_all_metrics(sample_pcs[:min_len].cuda(), ref_pcs[:min_len].cuda(), batch_size)
    print(res)


def nn1(sample_dir, ref_dir, batch_size=128, top=1, save_dir=None):
    print(sample_dir, ref_dir)
    sample_pcs, sample_path = sample_pc(sample_dir, return_path=True)
    ref_pcs, ref_path = sample_pc(ref_dir, return_path=True)
    sample_pcs, ref_pcs = torch.tensor(sample_pcs), torch.tensor(ref_pcs)
    res = compute_all_metrics(sample_pcs.cuda(), ref_pcs.cuda(), batch_size, save_dir=save_dir, sample_path=sample_path, ref_path=ref_path, top=top)
    print(res)
    
    
if __name__ == '__main__':
    sample_dir = f"output/PivotMesh"
    ref_dir = f"{os.environ['HOME']}/data/objaverse-lp-500/val-500"
    main(sample_dir, ref_dir)
    