from math import ceil

import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F

from torchtyping import TensorType

from pytorch_custom_utils import save_load

from beartype import beartype
from beartype.typing import Union, Tuple, Callable, Optional, List, Any

from einops import rearrange, repeat, pack

from x_transformers import Decoder
from x_transformers.x_transformers import LayerIntermediates

from x_transformers.autoregressive_wrapper import (
    eval_decorator,
    top_k,
)

from classifier_free_guidance_pytorch import (
    classifier_free_guidance,
    TextEmbeddingReturner
)

from tqdm import tqdm


# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(it):
    return it[0]

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def is_empty(l):
    return len(l) == 0

def is_tensor_empty(t: Tensor):
    return t.numel() == 0

def set_module_requires_grad_(
    module: Module,
    requires_grad: bool
):
    for param in module.parameters():
        param.requires_grad = requires_grad

def l1norm(t):
    return F.normalize(t, dim = -1, p = 1)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def safe_cat(tensors, dim):
    tensors = [*filter(exists, tensors)]

    if len(tensors) == 0:
        return None
    elif len(tensors) == 1:
        return first(tensors)

    return torch.cat(tensors, dim = dim)

def pad_at_dim(t, padding, dim = -1, value = 0):
    ndim = t.ndim
    right_dims = (ndim - dim - 1) if dim >= 0 else (-dim - 1)
    zeros = (0, 0) * right_dims
    return F.pad(t, (*zeros, *padding), value = value)

def pad_to_length(t, length, dim = -1, value = 0, right = True):
    curr_length = t.shape[dim]
    remainder = length - curr_length

    if remainder <= 0:
        return t

    padding = (0, remainder) if right else (remainder, 0)
    return pad_at_dim(t, padding, dim = dim, value = value)


@save_load()
class MeshTransformer(Module):
    @beartype
    def __init__(
        self,
        autoencoder,
        *,
        dim: Union[int, Tuple[int, int]] = 512,
        max_seq_len = 8192,
        flash_attn = True,
        attn_depth = 12,
        attn_dim_head = 64,
        attn_heads = 16,
        attn_kwargs: dict = dict(
            ff_glu = True,
            num_mem_kv = 4
        ),
        dropout = 0.,
        pad_id = -1,
        condition_on_text = False,
        text_condition_model_types = ('t5',),
        text_condition_cond_drop_prob = 0.25,
        mode = 'vertices',
    ):
        super().__init__()

        self.autoencoder = autoencoder
        set_module_requires_grad_(autoencoder, False)

        self.codebook_size = autoencoder.codebook_size
        self.num_quantizers = autoencoder.num_quantizers

        self.sos_token = nn.Parameter(torch.randn(dim))
        self.eos_token_id = self.codebook_size

        # they use axial positional embeddings
        self.mode = mode

        assert divisible_by(max_seq_len, self.num_quantizers), f'max_seq_len ({max_seq_len}) must be divisible by {self.num_quantizers}'

        self.token_embed = nn.Embedding(self.codebook_size + 1, dim)

        self.quantize_level_embed = nn.Parameter(torch.randn(self.num_quantizers, dim))
        
        if self.mode == 'vertices':
            self.vertex_embed = nn.Parameter(torch.randn(3, dim))

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim)

        self.max_seq_len = max_seq_len

        # text condition

        self.condition_on_text = condition_on_text
        self.conditioner = None

        cross_attn_dim_context = None

        if condition_on_text:
            self.conditioner = TextEmbeddingReturner(
                model_types = text_condition_model_types,
                cond_drop_prob = text_condition_cond_drop_prob
            )
            cross_attn_dim_context = self.conditioner.dim_latent

        # main autoregressive attention network

        self.decoder = Decoder(
            dim = dim,
            depth = attn_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            attn_flash = flash_attn,
            attn_dropout = dropout,
            ff_dropout = dropout,
            cross_attend = condition_on_text,
            cross_attn_dim_context = cross_attn_dim_context,
            **attn_kwargs
        )

        self.to_logits = nn.Linear(dim, self.codebook_size + 1)

        # padding id
        # force the autoencoder to use the same pad_id given in transformer

        self.pad_id = pad_id
        autoencoder.pad_id = pad_id

    @property
    def device(self):
        return next(self.parameters()).device

    @beartype
    @torch.no_grad()
    def embed_texts(self, texts: Union[str, List[str]]):
        single_text = not isinstance(texts, list)
        if single_text:
            texts = [texts]

        assert exists(self.conditioner)
        text_embeds = self.conditioner.embed_texts(texts).detach()

        if single_text:
            text_embeds = text_embeds[0]

        return text_embeds

    @eval_decorator
    @torch.no_grad()
    @beartype
    def generate(
        self,
        prompt: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        filter_logits_fn: Callable = top_k,
        filter_kwargs: dict = dict(),
        temperature = 1.,
        return_codes = False,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        cond_scale = 1.,
        cache_kv = True,
        max_seq_len = None,
        face_coords_to_file: Optional[Callable[[Tensor], Any]] = None
    ):
        max_seq_len = default(max_seq_len, self.max_seq_len)

        if exists(prompt):
            assert not exists(batch_size)

            prompt = rearrange(prompt, 'b ... -> b (...)')
            assert prompt.shape[-1] <= self.max_seq_len

            batch_size = prompt.shape[0]

        if self.condition_on_text:
            assert exists(texts) ^ exists(text_embeds), '`text` or `text_embeds` must be passed in if `condition_on_text` is set to True'
            if exists(texts):
                text_embeds = self.embed_texts(texts)

            batch_size = default(batch_size, text_embeds.shape[0])

        batch_size = default(batch_size, 1)

        codes = default(prompt, torch.empty((batch_size, 0), dtype = torch.long, device = self.device))

        curr_length = codes.shape[-1]

        cache = (None, None)

        for i in tqdm(range(curr_length, max_seq_len)):
            # [sos] v1([q1] [q2] [q1] [q2] [q1] [q2]) v2([q1] [q2] [q1] [q2] [q1] [q2]) -> 0 1 2 3 4 5 6 7 8 9 10 11 12 -> F v1(F F F F F T) v2(F F F F F T)

            # div_num = self.num_quantizers * 3 if self.mode == 'vertices' else self.num_quantizers
            # can_eos = i != 0 and divisible_by(i, div_num)  # only allow for eos to be decoded at the end of each face, defined as 3 vertices with D residual VQ codes

            #TODO: implement can_eos for pivotGPT
            # if no self.pad_token_id, can_eos = False
            # if yes, the mesh code length should be divided by 6

            output = self.forward_on_codes(
                codes,
                text_embeds = text_embeds,
                return_loss = False,
                return_cache = cache_kv,
                append_eos = False,
                cond_scale = cond_scale,
                cfg_routed_kwargs = dict(
                    cache = cache
                )
            )

            if cache_kv:
                logits, cache = output

                if cond_scale == 1.:
                    cache = (cache, None)
            else:
                logits = output

            logits = logits[:, -1]

            # if not can_eos:
            #     logits[:, -1] = -torch.finfo(logits.dtype).max

            filtered_logits = filter_logits_fn(logits, **filter_kwargs)
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sample = torch.multinomial(probs, 1)
            codes, _ = pack([codes, sample], 'b *')

            # check for all rows to have [eos] to terminate

            is_eos_codes = (codes == self.eos_token_id)

            if is_eos_codes.any(dim = -1).all():
                break

        # mask out to padding anything after the first eos

        mask = is_eos_codes.float().cumsum(dim = -1) >= 1
        codes = codes.masked_fill(mask, self.pad_id)
        
        # remove the pivot points if exist
        if (codes == self.eos_token_id - 1).any():
            mesh_codes = torch.ones_like(codes) * self.pad_id            
            pivot_lens = torch.cumsum(codes == self.eos_token_id - 1, dim=-1).argmax(dim=-1)
            for i in range(codes.shape[0]):
                mesh_len = codes.shape[1] - pivot_lens[i] -1
                mesh_codes[i, :mesh_len] = codes[i, pivot_lens[i]+1:]
            
            mesh_len = torch.nonzero((mesh_codes == self.pad_id).all(dim=0))[0] - 1
            codes = mesh_codes[:, :mesh_len] 

        # remove a potential extra token from eos, if breaked early

        code_len = codes.shape[-1]
        round_down_code_len = code_len // (3 * self.num_quantizers) * (3 * self.num_quantizers)
        codes = codes[:, :round_down_code_len]

        # early return of raw residual quantizer codes

        if return_codes:
            codes = rearrange(codes, 'b (n q) -> b n q', q = self.num_quantizers)
            return codes

        self.autoencoder.eval()
        face_coords, face_mask = self.autoencoder.decode_from_codes_to_faces(codes)

        if not exists(face_coords_to_file):
            return face_coords, face_mask

        files = [face_coords_to_file(coords[mask]) for coords, mask in zip(face_coords, face_mask)]
        return files

    def forward(
        self,
        *,
        vertices:       Optional[TensorType['b', 'nv', 3, float]] = None,
        pivot_mask:       Optional[TensorType['b', 'nv', bool]] = None,
        faces:          Optional[TensorType['b', 'nf', 3, int]] = None,
        face_edges:     Optional[TensorType['b', 'e', 2, int]] = None,
        codes:          Optional[Tensor] = None,
        cache:          Optional[LayerIntermediates] = None,
        **kwargs
    ):
        if not exists(codes):
            if pivot_mask is not None:
                codes = self.autoencoder.tokenize(
                    vertices = vertices,
                    faces = faces,
                    face_edges = face_edges,
                    pivot_mask = pivot_mask,
                )
            else:
                codes = self.autoencoder.tokenize(
                    vertices = vertices,
                    faces = faces,
                    face_edges = face_edges,
                )

        return self.forward_on_codes(codes, cache = cache, **kwargs)

    @classifier_free_guidance
    def forward_on_codes(
        self,
        codes = None,
        return_loss = True,
        return_cache = False,
        append_eos = True,
        cache = None,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        cond_drop_prob = 0.
    ):
        # handle text conditions

        attn_context_kwargs = dict()

        if self.condition_on_text:
            assert exists(texts) ^ exists(text_embeds), '`text` or `text_embeds` must be passed in if `condition_on_text` is set to True'

            if exists(texts):
                text_embeds = self.conditioner.embed_texts(texts)

            if exists(codes):
                assert text_embeds.shape[0] == codes.shape[0], 'batch size of texts or text embeddings is not equal to the batch size of the mesh codes'

            _, maybe_dropped_text_embeds = self.conditioner(
                text_embeds = text_embeds,
                cond_drop_prob = cond_drop_prob
            )

            attn_context_kwargs = dict(
                context = maybe_dropped_text_embeds.embed,
                context_mask = maybe_dropped_text_embeds.mask
            )

        # take care of codes that may be flattened

        if codes.ndim > 2:
            codes = rearrange(codes, 'b ... -> b (...)')

        # get some variable

        batch, seq_len, device = *codes.shape, codes.device

        assert seq_len <= self.max_seq_len, f'received codes of length {seq_len} but needs to be less than or equal to set max_seq_len {self.max_seq_len}'

        # auto append eos token

        if append_eos:
            assert exists(codes)

            code_lens = ((codes == self.pad_id).cumsum(dim = -1) == 0).sum(dim = -1)

            codes = F.pad(codes, (0, 1), value = 0)

            batch_arange = torch.arange(batch, device = device)

            batch_arange = rearrange(batch_arange, '... -> ... 1')
            code_lens = rearrange(code_lens, '... -> ... 1')

            codes[batch_arange, code_lens] = self.eos_token_id

        # if returning loss, save the labels for cross entropy

        if return_loss:
            assert seq_len > 0
            codes, labels = codes[:, :-1], codes

        # token embed (each residual VQ id)

        codes = codes.masked_fill(codes == self.pad_id, 0)
        codes = self.token_embed(codes)

        # codebook embed + absolute positions

        seq_arange = torch.arange(codes.shape[-2], device = device)

        codes = codes + self.abs_pos_emb(seq_arange)

        # embedding for quantizer level

        code_len = codes.shape[1]

        level_embed = repeat(self.quantize_level_embed, 'q d -> (r q) d', r = ceil(code_len / self.num_quantizers))
        codes = codes + level_embed[:code_len]

        # # embedding for each vertex
        if self.mode == 'vertices':
            vertex_embed = repeat(self.vertex_embed, 'nv d -> (r nv q) d', r = ceil(code_len / (3 * self.num_quantizers)), q = self.num_quantizers)
            codes = codes + vertex_embed[:code_len]

        # auto prepend sos token

        sos = repeat(self.sos_token, 'd -> b d', b = batch)
        codes, _ = pack([sos, codes], 'b * d')

        # attention

        attended, intermediates_with_cache = self.decoder(
            codes,
            cache = cache,
            return_hiddens = True,
            **attn_context_kwargs
        )

        # logits

        logits = self.to_logits(attended)

        if not return_loss:
            if not return_cache:
                return logits

            return logits, intermediates_with_cache

        # loss

        ce_loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.pad_id
        )

        return ce_loss
