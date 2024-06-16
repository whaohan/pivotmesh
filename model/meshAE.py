from functools import partial
from math import pi
import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast
from torchtyping import TensorType
from pytorch_custom_utils import save_load
from beartype import beartype
from beartype.typing import Tuple, Optional
from einops import rearrange, repeat, reduce, pack
from einops.layers.torch import Rearrange
from x_transformers import Encoder
from x_transformers.x_transformers import AbsolutePositionalEmbedding
from vector_quantize_pytorch import (
    ResidualVQ,
    ResidualLFQ
)
from meshgpt_pytorch.data import derive_face_edges_from_faces
from meshgpt_pytorch.meshgpt_pytorch import (
    discretize, get_derived_face_features,
    scatter_mean, default, pad_at_dim, exists, gaussian_blur_1d,
    undiscretize, first,
)
from torch_geometric.nn.conv import SAGEConv
import torch.nn.functional as F

@save_load()
class MeshAutoencoder(Module):
    @beartype
    def __init__(
        self,
        encoder_depth = 12,
        encoder_heads = 8,
        encoder_dim = 512,
        decoder_fine_dim = 192,
        num_discrete_coors = 128,
        coor_continuous_range: Tuple[float, float] = (-1., 1.),
        dim_coor_embed = 64,
        num_discrete_area = 128,
        dim_area_embed = 16,
        num_discrete_normals = 128,
        dim_normal_embed = 64,
        num_discrete_angle = 128,
        dim_angle_embed = 16,
        dim_codebook = 192,
        num_quantizers = 2,           # or 'D' in the paper
        codebook_size = 16384,        # they use 16k, shared codebook between layers
        use_residual_lfq = True,      # whether to use the latest lookup-free quantization
        rq_kwargs: dict = dict(
            quantize_dropout = True,
            quantize_dropout_cutoff_index = 1,
            quantize_dropout_multiple_of = 1,
        ),
        rvq_kwargs: dict = dict(
            kmeans_init = True,
            threshold_ema_dead_code = 2,
        ),
        rlfq_kwargs: dict = dict(
            frac_per_sample_entropy = 1.
        ),
        rvq_stochastic_sample_codes = True,
        commit_loss_weight = 0.1,
        bin_smooth_blur_sigma = 0.4,  # they blur the one hot discretized coordinate positions
        pad_id = -1,
        checkpoint_quantizer = False,
        patch_size = 1,
        dropout = 0.,
        quant_face = False,
        use_abs_pos = False,
        graph_layers = 1,
    ):
        super().__init__()

        # main face coordinate embedding

        self.num_discrete_coors = num_discrete_coors
        self.coor_continuous_range = coor_continuous_range

        self.discretize_face_coords = partial(discretize, num_discrete = num_discrete_coors, continuous_range = coor_continuous_range)
        self.coor_embed = nn.Embedding(num_discrete_coors, dim_coor_embed)

        # derived feature embedding

        self.discretize_angle = partial(discretize, num_discrete = num_discrete_angle, continuous_range = (0., pi))
        self.angle_embed = nn.Embedding(num_discrete_angle, dim_angle_embed)

        lo, hi = coor_continuous_range
        self.discretize_area = partial(discretize, num_discrete = num_discrete_area, continuous_range = (0., (hi - lo) ** 2))
        self.area_embed = nn.Embedding(num_discrete_area, dim_area_embed)

        self.discretize_normals = partial(discretize, num_discrete = num_discrete_normals, continuous_range = coor_continuous_range)
        self.normal_embed = nn.Embedding(num_discrete_normals, dim_normal_embed)

        # initial dimension

        init_dim = dim_coor_embed * 9 + dim_angle_embed * 3 + dim_normal_embed * 3 + dim_area_embed
        
        # patchify related
        
        self.patch_size = patch_size
        self.project_in = nn.Linear(init_dim * patch_size, encoder_dim)
        
        # graph init
        self.graph_layers = graph_layers
        if self.graph_layers > 0:
            sageconv_kwargs = dict(
                normalize = True,
                project = True,
            )
            self.init_sage_conv = SAGEConv(encoder_dim, encoder_dim, **sageconv_kwargs)
            self.init_encoder_act_and_norm = nn.Sequential(
                nn.SiLU(),
                nn.LayerNorm(encoder_dim)
            )
            self.graph_encoders = nn.ModuleList([])

        for _ in range(graph_layers - 1):
            sage_conv = SAGEConv(
                encoder_dim,
                encoder_dim,
                **sageconv_kwargs
            )
            self.graph_encoders.append(sage_conv)
        
        # transformer encoder
        # NOTE: currently no positional embedding
        
        self.encoder = Encoder(
            dim = encoder_dim,
            depth = encoder_depth,
            heads = encoder_heads,
            attn_flash = True,
            attn_dropout = dropout,
            ff_dropout = dropout,
        )

        # residual quantization

        self.codebook_size = codebook_size + 1      # add 1 for pad token between pivots and codes
        self.num_quantizers = num_quantizers
        self.pad_token_id = codebook_size

        self.sim_quant = quant_face
        if self.sim_quant:
            dim_codebook2 = dim_codebook
            self.project_dim_codebook = nn.Linear(encoder_dim, dim_codebook)
        else:
            dim_codebook = dim_codebook
            dim_codebook2 = dim_codebook * 3
            self.project_dim_codebook = nn.Linear(encoder_dim, dim_codebook * 3)

        if use_residual_lfq:
            self.quantizer = ResidualLFQ(
                dim = dim_codebook,
                num_quantizers = num_quantizers,
                codebook_size = codebook_size,
                commitment_loss_weight = 1.,
                **rlfq_kwargs,
                **rq_kwargs
            )
        else:
            self.quantizer = ResidualVQ(
                dim = dim_codebook,
                num_quantizers = num_quantizers,
                codebook_size = codebook_size,
                shared_codebook = True,
                commitment_weight = 1.,
                stochastic_sample_codes = rvq_stochastic_sample_codes,
                **rvq_kwargs,
                **rq_kwargs
            )

        self.checkpoint_quantizer = checkpoint_quantizer # whether to memory checkpoint the quantizer

        self.pad_id = pad_id # for variable lengthed faces, padding quantized ids will be set to this value

        # decoder
        
        self.init_decoder = nn.Sequential(
            nn.Linear(dim_codebook2, encoder_dim),
            nn.SiLU(),
            nn.LayerNorm(encoder_dim),
        )

        self.decoder_coarse = Encoder(
            dim = encoder_dim,
            depth = encoder_depth // 2,
            heads = encoder_heads,
            attn_flash = True,
            attn_dropout = dropout,
            ff_dropout = dropout,
        )
        
        self.coarse_to_fine = nn.Linear(encoder_dim, decoder_fine_dim * 3)

        self.decoder_fine = Encoder(
            dim = decoder_fine_dim,
            depth = encoder_depth // 2,
            heads = encoder_heads,
            attn_flash = True,
            attn_dropout = dropout,
            ff_dropout = dropout,
        )

        self.to_coor_logits = nn.Sequential(
            nn.Linear(decoder_fine_dim, patch_size * num_discrete_coors * 3),
            Rearrange('b (nf nv) (v c) -> b nf (nv v) c', nv = 3, v = 3)
            # Rearrange('b np (p v c) -> b (np p) v c', v = 9, p = patch_size)
        )

        # loss related

        self.commit_loss_weight = commit_loss_weight
        self.bin_smooth_blur_sigma = bin_smooth_blur_sigma

    @beartype
    def encode(
        self,
        *,
        vertices:         TensorType['b', 'nv', 3, float],
        faces:            TensorType['b', 'nf', 3, int],
        face_edges:       TensorType['b', 'e', 2, int],
        face_mask:        TensorType['b', 'nf', bool],
        face_edges_mask:  TensorType['b', 'e', bool],
        return_face_coordinates = False
    ):
        """
        einops:
        b - batch
        nf - number of faces
        nv - number of vertices (3)
        c - coordinates (3)
        d - embed dim
        """

        batch, num_vertices, num_coors, device = *vertices.shape, vertices.device
        _, num_faces, _ = faces.shape

        face_without_pad = faces.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1'), 0)

        faces_vertices = repeat(face_without_pad, 'b nf nv -> b nf nv c', c = num_coors)
        vertices = repeat(vertices, 'b nv c -> b nf nv c', nf = num_faces)

        # continuous face coords

        face_coords = vertices.gather(-2, faces_vertices)

        # compute derived features and embed

        derived_features = get_derived_face_features(face_coords)

        discrete_angle = self.discretize_angle(derived_features['angles'])
        angle_embed = self.angle_embed(discrete_angle)

        discrete_area = self.discretize_area(derived_features['area'])
        area_embed = self.area_embed(discrete_area)

        discrete_normal = self.discretize_normals(derived_features['normals'])
        normal_embed = self.normal_embed(discrete_normal)

        # discretize vertices for face coordinate embedding

        discrete_face_coords = self.discretize_face_coords(face_coords)
        discrete_face_coords = rearrange(discrete_face_coords, 'b nf nv c -> b nf (nv c)') # 9 coordinates per face

        face_coor_embed = self.coor_embed(discrete_face_coords)
        face_coor_embed = rearrange(face_coor_embed, 'b nf c d -> b nf (c d)')

        # combine all features and project into model dimension

        face_embed, _ = pack([face_coor_embed, angle_embed, area_embed, normal_embed], 'b nf *')
        
        # patchify faces: [b nf d] -> [b nf//patch_size d]
        
        face_embed = rearrange(face_embed, 'b (num_patch patch_size) d -> b num_patch (patch_size d)', 
                               patch_size = self.patch_size)
        face_embed = self.project_in(face_embed)
        
        face_mask = rearrange(face_mask, 'b (num_patch patch_size) -> b num_patch patch_size', 
                              patch_size = self.patch_size).float().mean(dim=-1).bool()
        
        # face_embed = face_embed * face_mask.unsqueeze(-1)

        # init graph 
        
        if self.graph_layers > 0:
            orig_face_embed_shape = face_embed.shape[:2]
            face_embed = face_embed[face_mask]
            
            face_index_offsets = reduce(face_mask.long(), 'b nf -> b', 'sum')
            face_index_offsets = F.pad(face_index_offsets.cumsum(dim = 0), (1, -1), value = 0)
            face_index_offsets = rearrange(face_index_offsets, 'b -> b 1 1')
            
            face_edges = face_edges + face_index_offsets
            face_edges = face_edges[face_edges_mask]
            face_edges = rearrange(face_edges, 'be ij -> ij be')
            
            
            face_embed = self.init_sage_conv(face_embed, face_edges)
            face_embed = self.init_encoder_act_and_norm(face_embed)

            for conv in self.graph_encoders:
                face_embed = conv(face_embed, face_edges)
            
            shape = (*orig_face_embed_shape, face_embed.shape[-1])
            face_embed = face_embed.new_zeros(shape).masked_scatter(rearrange(face_mask, '... -> ... 1'), face_embed)  
        
        # encode face embeddings
        face_embed = self.encoder(face_embed, mask=face_mask)
        
        if not return_face_coordinates:
            return face_embed

        return face_embed, discrete_face_coords

    @beartype
    def quantize_old(
        self,
        *,
        faces: TensorType['b', 'nf', 3, int],
        face_mask: TensorType['b', 'n', bool],
        face_embed: TensorType['b', 'nf', 'd', float],
        pivot_mask: Optional[TensorType['b', 'nv', bool]] = None,
        pad_id = None,
        rvq_sample_codebook_temp = 1.
    ):
        pad_id = default(pad_id, self.pad_id)
        batch, num_faces, device = *faces.shape[:2], faces.device

        max_vertex_index = faces.amax()
        num_vertices = int(max_vertex_index.item() + 1)

        face_embed = self.project_dim_codebook(face_embed)
        face_embed = rearrange(face_embed, 'b nf (nv d) -> b nf nv d', nv = 3)

        vertex_dim = face_embed.shape[-1]
        vertices = torch.zeros((batch, num_vertices, vertex_dim), 
                               device = device, dtype=face_embed.dtype)

        # create pad vertex, due to variable lengthed faces

        pad_vertex_id = num_vertices
        vertices = pad_at_dim(vertices, (0, 1), dim = -2, value = 0.)

        faces = faces.masked_fill(~rearrange(face_mask, 'b n -> b n 1'), pad_vertex_id)

        # prepare for scatter mean

        faces_with_dim = repeat(faces, 'b nf nv -> b (nf nv) d', d = vertex_dim)

        face_embed = rearrange(face_embed, 'b ... d -> b (...) d')

        # scatter mean

        averaged_vertices = scatter_mean(vertices, faces_with_dim, face_embed, dim = -2)

        # mask out null vertex token

        mask = torch.ones((batch, num_vertices + 1), device = device, dtype = torch.bool)
        mask[:, -1] = False

        # rvq specific kwargs

        quantize_kwargs = dict(mask = mask)

        if isinstance(self.quantizer, ResidualVQ):
            quantize_kwargs.update(sample_codebook_temp = rvq_sample_codebook_temp)

        # a quantize function that makes it memory checkpointable

        def quantize_wrapper_fn(inp):
            unquantized, quantize_kwargs = inp
            return self.quantizer(unquantized, **quantize_kwargs)

        # maybe checkpoint the quantize fn

        if self.checkpoint_quantizer:
            quantize_wrapper_fn = partial(checkpoint, quantize_wrapper_fn, use_reentrant = False)

        # residual VQ

        quantized, codes, commit_loss = quantize_wrapper_fn((averaged_vertices, quantize_kwargs))

        # gather quantized vertexes back to faces for decoding
        # now the faces have quantized vertices

        face_embed_output = quantized.gather(-2, faces_with_dim)
        face_embed_output = rearrange(face_embed_output, 'b (nf nv) d -> b nf (nv d)', nv = 3)

        # vertex codes also need to be gathered to be organized by face sequence
        # for autoregressive learning

        faces_with_quantized_dim = repeat(faces, 'b nf nv -> b (nf nv) q', q = self.num_quantizers)
        codes_output = codes.gather(-2, faces_with_quantized_dim)

        # make sure codes being outputted have this padding

        face_mask = repeat(face_mask, 'b nf -> b (nf nv) 1', nv = 3)
        codes_output = codes_output.masked_fill(~face_mask, self.pad_id)
        
        if pivot_mask is not None:
            # collect vertex codes of pivot points
            pivot_ind = pivot_mask
            pivot_mask = pivot_ind != self.pad_id
            pivot_ind[~pivot_mask] = 0

            pivot_ind = repeat(pivot_ind, 'b nv -> b nv q', q = self.num_quantizers)
            pivot_codes = codes.gather(-2, pivot_ind)
            pivot_codes[~pivot_mask] = self.pad_token_id

            # construct the final codes with pivot points
            pivot_lens, face_lens = pivot_mask.sum(dim=1), face_mask.sum(dim=1)
            max_len = (pivot_lens + face_lens).max() + 1
            final_codes = torch.ones(codes_output.shape[0], max_len, 
                                    codes_output.shape[2], dtype=torch.long).to(codes_output.device) * -1
            for i in range(pivot_codes.shape[0]):
                
                final_codes[i, :pivot_lens[i]] = pivot_codes[i, :pivot_lens[i]]
                final_codes[i, pivot_lens[i]] = self.pad_token_id
                end = min(max_len - pivot_lens[i] - 1, codes_output.shape[1])
                final_codes[i, pivot_lens[i] + 1: pivot_lens[i] + 1 + end] = codes_output[i, :end]
                
            codes_output = final_codes
            
        
        # output quantized, codes, as well as commitment loss

        return face_embed_output, codes_output, commit_loss

    @beartype
    def quantize_vert(
        self,
        *,
        faces: TensorType['b', 'nf', 3, int],
        face_mask: TensorType['b', 'n', bool],
        face_embed: TensorType['b', 'nf', 'd', float],
        pad_id = None,
        rvq_sample_codebook_temp = 1.
    ):
        pad_id = default(pad_id, self.pad_id)
        # batch, num_faces, device = *faces.shape[:2], faces.device

        # max_vertex_index = faces.amax()
        # num_vertices = int(max_vertex_index.item() + 1)

        face_embed = self.project_dim_codebook(face_embed)
        face_embed = rearrange(face_embed, 'b nf (nv d) -> b nf nv d', nv = 3)

        # vertex_dim = face_embed.shape[-1]
        # vertices = torch.zeros((batch, num_vertices, vertex_dim), device = device)

        # # create pad vertex, due to variable lengthed faces

        # pad_vertex_id = num_vertices
        # vertices = pad_at_dim(vertices, (0, 1), dim = -2, value = 0.)

        # faces = faces.masked_fill(~rearrange(face_mask, 'b n -> b n 1'), pad_vertex_id)

        # # prepare for scatter mean

        # faces_with_dim = repeat(faces, 'b nf nv -> b (nf nv) d', d = vertex_dim)

        face_embed = rearrange(face_embed, 'b ... d -> b (...) d')

        # # scatter mean

        # averaged_vertices = scatter_mean(vertices, faces_with_dim, face_embed, dim = -2)

        # mask out null vertex token

        # mask = torch.ones((batch, num_vertices + 1), device = device, dtype = torch.bool)
        # mask[:, -1] = False

        # rvq specific kwargs

        quantize_kwargs = dict()
        # quantize_kwargs = dict(mask = mask)

        if isinstance(self.quantizer, ResidualVQ):
            quantize_kwargs.update(sample_codebook_temp = rvq_sample_codebook_temp)

        # a quantize function that makes it memory checkpointable

        def quantize_wrapper_fn(inp):
            unquantized, quantize_kwargs = inp
            return self.quantizer(unquantized, **quantize_kwargs)

        # maybe checkpoint the quantize fn

        if self.checkpoint_quantizer:
            quantize_wrapper_fn = partial(checkpoint, quantize_wrapper_fn, use_reentrant = False)

        # residual VQ

        quantized, codes, commit_loss = quantize_wrapper_fn((face_embed, quantize_kwargs))
        # quantized, codes, commit_loss = quantize_wrapper_fn((averaged_vertices, quantize_kwargs))

        # gather quantized vertexes back to faces for decoding
        # now the faces have quantized vertices

        # face_embed_output = quantized.gather(-2, faces_with_dim)
        # face_embed_output = rearrange(face_embed_output, 'b (nf nv) d -> b nf (nv d)', nv = 3)
        face_embed_output = rearrange(quantized, 'b (nf nv) d -> b nf (nv d)', nv = 3)

        # vertex codes also need to be gathered to be organized by face sequence
        # for autoregressive learning

        # faces_with_quantized_dim = repeat(faces, 'b nf nv -> b (nf nv) q', q = self.num_quantizers)
        # codes_output = codes.gather(-2, faces_with_quantized_dim)
        codes_output = codes

        # make sure codes being outputted have this padding

        face_mask = repeat(face_mask, 'b nf -> b (nf nv) 1', nv = 3)
        codes_output = codes_output.masked_fill(~face_mask, self.pad_id)

        # output quantized, codes, as well as commitment loss

        return face_embed_output, codes_output, commit_loss


    @beartype
    def quantize_face(
        self,
        *,
        faces: TensorType['b', 'nf', 3, int],
        face_mask: TensorType['b', 'n', bool],
        face_embed: TensorType['b', 'nf', 'd', float],
        pad_id = None,
        rvq_sample_codebook_temp = 1.
    ):
        # project face embeddings to codebook embeddings
        
        # maybe commented
        face_embed = self.project_dim_codebook(face_embed)
        face_embed = face_embed.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1'), 0.)
        
        # rvq specific kwargs

        quantize_kwargs = dict(mask = face_mask)
        # quantize_kwargs = dict()

        if isinstance(self.quantizer, ResidualVQ):
            quantize_kwargs.update(sample_codebook_temp = rvq_sample_codebook_temp)

        # a quantize function that makes it memory checkpointable

        def quantize_wrapper_fn(inp):
            unquantized, quantize_kwargs = inp
            return self.quantizer(unquantized, **quantize_kwargs)

        # maybe checkpoint the quantize fn

        if self.checkpoint_quantizer:
            quantize_wrapper_fn = partial(checkpoint, quantize_wrapper_fn, use_reentrant = False)

        # residual VQ

        quantized, codes, commit_loss = quantize_wrapper_fn((face_embed, quantize_kwargs))

        # make sure codes being outputted have this padding

        face_mask = rearrange(face_mask, 'b nf -> b nf 1')
        codes_output = codes.masked_fill(~face_mask, self.pad_id)

        # output quantized, codes, as well as commitment loss

        return quantized, codes_output, commit_loss

    @beartype
    def decode(
        self,
        quantized: TensorType['b', 'n', 'd', float],
        face_mask:  TensorType['b', 'n', bool]
    ):
        conv_face_mask = rearrange(face_mask, 'b n -> b n 1')
        vertice_mask = repeat(face_mask, 'b nf -> b (nf nv)', nv = 3)
        x = quantized
        x = x.masked_fill(~conv_face_mask, 0.)

        x = self.init_decoder(x)
        # x = self.decoder(x)
        x = self.decoder_coarse(x, mask = face_mask)
        x = self.coarse_to_fine(x)
        x = rearrange(x, 'b nf (nv d) -> b (nf nv) d', nv=3)
        x = self.decoder_fine(x, mask = vertice_mask)
        
        return x

    @beartype
    @torch.no_grad()
    def decode_from_codes_to_faces(
        self,
        codes: Tensor,
        face_mask: Optional[TensorType['b', 'n', bool]] = None,
        return_discrete_codes = False
    ):
        if self.sim_quant:
            code_len = codes.shape[1] // self.num_quantizers * self.num_quantizers
        else:
            code_len = codes.shape[1] // 3 * 3
            
        codes = codes[:, :code_len]

        codes = rearrange(codes, 'b ... -> b (...)')

        if not exists(face_mask):
            if self.sim_quant:
                face_mask = reduce(codes != self.pad_id, 'b (nf q) -> b nf', 'all', q = self.num_quantizers)
            else:
                face_mask = reduce(codes != self.pad_id, 'b (nf nv q) -> b nf', 'all', nv = 3, q = self.num_quantizers)

        # handle different code shapes

        codes = rearrange(codes, 'b (n q) -> b n q', q = self.num_quantizers)

        # decode

        quantized = self.quantizer.get_output_from_indices(codes)
        if not self.sim_quant:
            quantized = rearrange(quantized, 'b (nf nv) d -> b nf (nv d)', nv = 3)

        decoded = self.decode(
            quantized,
            face_mask = face_mask
        )


        # decoded = decoded.masked_fill(~face_mask[..., None], 0.)
        pred_face_coords = self.to_coor_logits(decoded)

        pred_face_coords = pred_face_coords.argmax(dim = -1)

        pred_face_coords = rearrange(pred_face_coords, '... (v c) -> ... v c', v = 3)

        # back to continuous space

        continuous_coors = undiscretize(
            pred_face_coords,
            num_discrete = self.num_discrete_coors,
            continuous_range = self.coor_continuous_range
        )

        # mask out with nan

        continuous_coors = continuous_coors.masked_fill(~rearrange(face_mask, 'b nf -> b nf 1 1'), float('nan'))

        if not return_discrete_codes:
            return continuous_coors, face_mask

        return continuous_coors, pred_face_coords, face_mask

    @torch.no_grad()
    def tokenize(self, vertices, faces, face_edges = None, pivot_mask = None, **kwargs):
        assert 'return_codes' not in kwargs

        inputs = [vertices, faces, face_edges]
        inputs = [*filter(exists, inputs)]
        ndims = {i.ndim for i in inputs}

        assert len(ndims) == 1
        batch_less = first(list(ndims)) == 2

        if batch_less:
            inputs = [rearrange(i, '... -> 1 ...') for i in inputs]

        input_kwargs = dict(zip(['vertices', 'faces', 'face_edges'], inputs))

        self.eval()

        codes = self.forward(
            **input_kwargs,
            pivot_mask = pivot_mask,
            return_codes = True,
            **kwargs
        )

        if batch_less:
            codes = rearrange(codes, '1 ... -> ...')

        return codes

    @beartype
    def forward(
        self,
        *,
        vertices:       TensorType['b', 'nv', 3, float],
        faces:          TensorType['b', 'nf', 3, int],
        face_edges:     Optional[TensorType['b', 'e', 2, int]] = None,
        pivot_mask:     Optional[TensorType['b', 'nv', bool]] = None,
        return_codes = False,
        return_loss_breakdown = False,
        return_recon_faces = False,
        only_return_recon_faces = False,
        rvq_sample_codebook_temp = 1.
    ):
        if not exists(face_edges):
            face_edges = derive_face_edges_from_faces(faces, pad_id = self.pad_id)

        num_faces, num_face_edges, device = faces.shape[1], face_edges.shape[1], faces.device

        face_mask = reduce(faces != self.pad_id, 'b nf c -> b nf', 'all')
        face_mask_patch = rearrange(face_mask, 'b (np p) -> b np p', p=self.patch_size).float().mean(dim=-1).bool()
        face_edges_mask = reduce(face_edges != self.pad_id, 'b e ij -> b e', 'all')

        encoded, face_coordinates = self.encode(
            vertices = vertices,
            faces = faces,
            face_edges = face_edges,
            face_edges_mask = face_edges_mask,
            face_mask = face_mask,
            return_face_coordinates = True
        )

        if self.sim_quant:
            quantized, codes, commit_loss = self.quantize_face(
                face_embed = encoded,
                faces = faces,
                face_mask = face_mask_patch,
                rvq_sample_codebook_temp = rvq_sample_codebook_temp
            )
        else:
            quantized, codes, commit_loss = self.quantize_old(
            # quantized, codes, commit_loss = self.quantize_vert(
                face_embed = encoded,
                faces = faces,
                face_mask = face_mask,
                pivot_mask = pivot_mask,
                rvq_sample_codebook_temp = rvq_sample_codebook_temp
            )
        
        # make sure the right data type    
        quantized = quantized.to(encoded.dtype)

        if return_codes:
            assert not return_recon_faces, 'cannot return reconstructed faces when just returning raw codes'
            return codes

        decode = self.decode(
            quantized,
            face_mask = face_mask_patch
        )

        pred_face_coords = self.to_coor_logits(decode)

        # compute reconstructed faces if needed

        if return_recon_faces or only_return_recon_faces:

            recon_faces = undiscretize(
                pred_face_coords.argmax(dim = -1),
                num_discrete = self.num_discrete_coors,
                continuous_range = self.coor_continuous_range,
            )

            recon_faces = rearrange(recon_faces, 'b nf (nv c) -> b nf nv c', nv = 3)
            face_mask = rearrange(face_mask, 'b nf -> b nf 1 1')
            recon_faces = recon_faces.masked_fill(~face_mask, float('nan'))
            face_mask = rearrange(face_mask, 'b nf 1 1 -> b nf')

        if only_return_recon_faces:
            return recon_faces, pred_face_coords, face_coordinates, face_mask

        # prepare for recon loss

        pred_face_coords = rearrange(pred_face_coords, 'b ... c -> b c (...)')
        face_coordinates = rearrange(face_coordinates, 'b ... -> b 1 (...)')

        # reconstruction loss on discretized coordinates on each face
        # they also smooth (blur) the one hot positions, localized label smoothing basically

        with autocast(enabled = False):
            pred_log_prob = pred_face_coords.log_softmax(dim = 1)

            target_one_hot = torch.zeros_like(pred_log_prob).scatter(1, face_coordinates, 1.)

            if self.bin_smooth_blur_sigma >= 0.:
                target_one_hot = gaussian_blur_1d(target_one_hot, sigma = self.bin_smooth_blur_sigma)

            # cross entropy with localized smoothing

            recon_losses = (-target_one_hot * pred_log_prob).sum(dim = 1)

            face_mask = repeat(face_mask, 'b nf -> b (nf r)', r = 9)
            recon_loss = recon_losses[face_mask].mean()

        # calculate total loss

        total_loss = recon_loss + \
                     commit_loss.sum() * self.commit_loss_weight

        # calculate loss breakdown if needed

        loss_breakdown = (recon_loss, commit_loss)

        # some return logic

        if not return_loss_breakdown:
            if not return_recon_faces:
                return total_loss

            return recon_faces, total_loss

        if not return_recon_faces:
            return total_loss, loss_breakdown

        return recon_faces, total_loss, loss_breakdown
    
    
    
if __name__ == '__main__':
    # init model
    meshAE = MeshAutoencoder()
    
    # mock input
    vertices = torch.randn(4, 99, 3)
    faces = torch.arange(99).reshape(33, 3)
    faces = repeat(faces, 'f c -> b f c', b=4)
    
    meshAE(vertices=vertices, faces=faces)