import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange, repeat, reduce, pack, unpack
from collections import namedtuple
import math
from taming.modules.vqvae.distrib import all_reduce
import torch.distributed as distributed
from torch.optim import Optimizer
from typing import Callable

LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'codebook_entropy', 'commitment', 'avg_probs'])

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def noop(*args, **kwargs):
    pass

def identity(t):
    return t

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]

def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype=torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)

def batched_sample_vectors(samples, num):
    return sample_vectors(samples, num)
    # return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim=0)], dim=0)

def pad_shape(shape, size, dim=0):
    return [size if i == dim else s for i, s in enumerate(shape)]

def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype=torch.long, device=x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)

def all_gather_variably_sized(x, sizes, dim=0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src=i, async_op=True)
        all_x.append(t)

    distributed.barrier()
    return all_x

def sample_vectors_distributed(local_samples, num):
    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim=0)

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src=0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim=0)
    out = torch.cat(all_samples, dim=0)

    return out

def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    e = rearrange(e, 'b d -> b 1 d')
    w = F.normalize(u + q, p = 2, dim = 1, eps = 1e-12).detach()

    return (
        e -
        2 * (e @ rearrange(w, 'b d -> b d 1') @ rearrange(w, 'b d -> b 1 d')) +
        2 * (e @ rearrange(u, 'b d -> b d 1').detach() @ rearrange(q, 'b d -> b 1 d').detach())
    )

def safe_div(num, den, eps = 1e-6):
    return num / den.clamp(min = eps)

def rotate_to(src, tgt):
    # rotation trick STE (https://arxiv.org/abs/2410.06424) to get gradients through VQ layer.
    src, inverse = pack_one(src, '* d')
    tgt, _ = pack_one(tgt, '* d')

    norm_src = src.norm(dim = -1, keepdim = True)
    norm_tgt = tgt.norm(dim = -1, keepdim = True)

    rotated_tgt = efficient_rotation_trick_transform(
        safe_div(src, norm_src),
        safe_div(tgt, norm_tgt),
        src
    ).squeeze()

    rotated = rotated_tgt * safe_div(norm_tgt, norm_src).detach()
    return unpack_one(rotated, inverse, '* d')


class SemAttnVQ(nn.Module):
    def __init__(self,
                 n_e,
                 e_dim,
                 e_type_num=16,
                 z_dim=-1,
                 beta=0.25,
                 decay=0.8,
                 eps=1.0e-5,
                 rotation_trick=False,
                 threshold_ema_dead_code=2,
                 reset_cluster_size=None,
                 use_ddp=True,
                 legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.e_type_num = e_type_num
        self.beta = beta
        self.legacy = legacy

        self.decay = decay
        self.eps = eps
        self.rotation_trick = rotation_trick
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.reset_cluster_size = default(reset_cluster_size, threshold_ema_dead_code)

        self.sample_fn = sample_vectors_distributed if use_ddp else batched_sample_vectors
        self.all_reduce_fn = all_reduce if use_ddp else noop

        self.sem = nn.Parameter(torch.randn(self.e_type_num, self.e_dim), requires_grad=True)
        # self.alpha = nn.Parameter(torch.ones(size=(self.e_type_num, 1)) * 0.1, requires_grad=True)
        self.q_proj = nn.Linear(self.e_dim, self.e_dim)
        self.k_proj = nn.Linear(self.e_dim, self.e_dim)
        self.v_proj = nn.Linear(self.e_dim, self.e_dim)

        # embed = torch.randn(n_e, e_dim)
        embed = nn.init.kaiming_uniform_(torch.empty(n_e, e_dim))
        self.register_buffer("embedding", embed)
        self.register_buffer('cluster_size', torch.zeros(n_e))
        self.register_buffer("proj_size", torch.zeros(n_e, e_type_num))
        self.register_buffer("perp_size", torch.zeros(n_e))
        embed_proj, embed_perp = self.init_embedding_proj_and_perp(embed.clone())
        self.register_buffer("embedding_proj", embed_proj)
        self.register_buffer("embedding_perp", embed_perp)

    @property
    def device(self):
        return self.embedding.device

    @property
    def norm_sem(self):
        return F.normalize(self.sem, dim=-1, p=2)

    @torch.no_grad()
    def init_embedding_proj_and_perp(self, embed):
        sim_es = embed @ self.norm_sem.t()
        embed_proj = sim_es.unsqueeze(-1) * self.norm_sem
        embed_perp = embed - embed_proj.sum(1)
        return embed_proj, embed_perp

    def get_norm_sem(self, reg=False, reg_term=None):
        if reg:
            q = self.q_proj(self.sem)  # [e_type_num, e_dim]
            k = self.k_proj(reg_term)  # [n, e_dim]
            v = self.v_proj(reg_term)
            attn_scores = torch.matmul(q, k.t())  # [e_type_num, n]
            attn_weights = F.softmax(attn_scores / np.sqrt(self.e_dim), dim=1)
            context = torch.matmul(attn_weights, v)  # [e_type_num, e_dim]
            sem_adjusted = self.sem + context
            norm_sem = F.normalize(sem_adjusted, dim=-1, p=2)
            return norm_sem
        else:
            return self.norm_sem

    def get_orth_loss(self, x=None, orth_lambda=1e-1):
        if x is None:
            x = self.norm_sem
        w_wt = x @ x.t()
        ones_diag = torch.ones_like(w_wt) - torch.eye(x.shape[0]).to(self.device)
        # loss = orth_lambda * (torch.norm(w_wt - diag, p="fro"))
        loss = orth_lambda * ((w_wt * ones_diag) ** 2).sum()
        return loss

    def replace(self, batch_samples, batch_mask):
        if not torch.any(batch_mask):
            return

        sampled = self.sample_fn(batch_samples, batch_mask.sum().item())
        self.embedding.data[batch_mask] = sampled.float()
        self.cluster_size.data[batch_mask] = self.reset_cluster_size

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, '... d -> (...) d')
        self.replace(batch_samples, batch_mask=expired_codes)

    def _quantize(self, z, prompts=None, no_grad=False):
        # import pdb; pdb.set_trace()
        z = z.detach()
        e = self.embedding.detach()
        prompts = z if prompts is None else prompts
        assert prompts.shape[-1] == self.e_dim
        norm_sem = self.get_norm_sem(reg=True, reg_term=prompts)

        sim_zs = F.linear(z, norm_sem)  # L x P
        # sim_zs = z @ norm_sem.t()
        z_proj = sim_zs.unsqueeze(-1) * norm_sem  # L x P x D
        z_perp = z - z_proj.sum(1)  # Complete the parts beyond k semantics

        sim_es = F.linear(e, norm_sem)  # K x P
        # sim_es = e @ norm_sem.t()
        e_proj = sim_es.unsqueeze(-1) * norm_sem  # K x P x D
        e_perp = e - e_proj.sum(1)  # K x D

        d_zproj_eproj = torch.abs(sim_es - sim_zs.unsqueeze(1))  # L x K x P
        min_encoding_indices_proj = torch.argmax(-d_zproj_eproj, dim=1)  # L x P
        d_zperp_eperp = torch.cdist(z_perp, e_perp)
        min_encoding_indices_perp = torch.argmax(-d_zperp_eperp, dim=1)  # L

        sim_zqs = torch.gather(sim_es.unsqueeze(0).repeat(z.shape[0], 1, 1), dim=1, index=min_encoding_indices_proj.unsqueeze(1)).squeeze()  # L x P
        z_q_proj = sim_zqs.unsqueeze(-1) * norm_sem  # L x P x D
        z_q_perp = F.one_hot(min_encoding_indices_perp, self.n_e).to(e_perp) @ e_perp
        z_q = torch.sum(z_q_proj, dim=1) + z_q_perp  # L x D

        if no_grad:
            return z_q.detach(), torch.cat([min_encoding_indices_proj, min_encoding_indices_perp.unsqueeze(-1)], dim=1)

        orth_loss = self.get_orth_loss(norm_sem)
        perp_loss = ((z_perp ** 2).mean() + (e_perp ** 2).mean()) * 1e-1  # Ablation experiments

        res = {
            'z_q': z_q,
            'norm_sem': norm_sem,
            'sim_zs': sim_zs,
            'z_perp': z_perp,
            'quant_loss': orth_loss + perp_loss,
            'min_encoding_indices_proj': min_encoding_indices_proj,
            'min_encoding_indices_perp': min_encoding_indices_perp,
        }

        return res

    def forward(self, z, prompts=None, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp == 1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits == False, "Only for interface compatible with Gumbel"
        assert return_logits == False, "Only for interface compatible with Gumbel"

        # reshape z -> (batch, height, width, channel) and flatten
        if z.ndim == 4:
            z = rearrange(z, 'b c h w -> b h w c').contiguous()
        elif z.ndim == 3:
            z = rearrange(z, 'b c h -> b h c').contiguous()
        else:
            raise ValueError(
                f"Invalid tensor dimension: expected 3 or 4 dimensions (got {z.ndim}D tensor)\n"
                f"Shape of z: {tuple(z.shape)}"
            )
        assert z.shape[-1] == self.e_dim
        z_flattened = z.view(-1, self.e_dim)

        quant_output = self._quantize(z_flattened, prompts)
        z_q = quant_output['z_q'].view(z.shape)
        quant_loss = quant_output['quant_loss']

        norm_sem = quant_output['norm_sem'].detach()
        sim_zs = quant_output['sim_zs'].detach()
        z_perp = quant_output['z_perp'].detach()
        min_encoding_indices_proj = quant_output['min_encoding_indices_proj'].detach()
        min_encoding_indices_perp = quant_output['min_encoding_indices_perp'].detach()
        min_encoding_indices = torch.cat([min_encoding_indices_proj, min_encoding_indices_perp.unsqueeze(-1)], dim=1)

        if self.training:
            proj_indices_onehot = F.one_hot(min_encoding_indices_proj, self.n_e).to(z_flattened.dtype)  # L x P x K
            proj_num_sum = (proj_indices_onehot.sum(0).t()).contiguous()
            proj_weighted_sum = (sim_zs.unsqueeze(-1) * proj_indices_onehot).sum(0).t()  # K x P
            z_proj_sum = (proj_weighted_sum.unsqueeze(-1) * norm_sem).contiguous()  # K x P x D

            perp_indices_onehot = F.one_hot(min_encoding_indices_perp, self.n_e).to(z_flattened.dtype)  # L x K
            perp_num_sum = (perp_indices_onehot.sum(0)).contiguous()  # K
            z_perp_sum = (perp_indices_onehot.t() @ z_perp).contiguous()  # K x D

            embed_onehot = torch.cat([proj_indices_onehot, perp_indices_onehot.unsqueeze(1)], dim=1).mean(dim=1)
            cluster_size = embed_onehot.sum(dim=0).contiguous()

            self.all_reduce_fn(proj_num_sum)
            self.all_reduce_fn(z_proj_sum)
            self.all_reduce_fn(perp_num_sum)
            self.all_reduce_fn(z_perp_sum)
            self.all_reduce_fn(cluster_size)

            self.proj_size.data.mul_(self.decay).add_(proj_num_sum, alpha=1 - self.decay)
            self.perp_size.data.mul_(self.decay).add_(perp_num_sum, alpha=1 - self.decay)
            self.embedding_proj.data.mul_(self.decay).add_(z_proj_sum, alpha=1 - self.decay)
            self.embedding_perp.data.mul_(self.decay).add_(z_perp_sum, alpha=1 - self.decay)
            self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

            n = self.proj_size.sum(0)
            proj_size = (self.proj_size + self.eps) / (n + self.n_e * self.eps) * n
            n = self.perp_size.sum()
            perp_size = (self.perp_size + self.eps) / (n + self.n_e * self.eps) * n
            embed_normalized = (self.embedding_proj / proj_size.unsqueeze(-1)).sum(1) + self.embedding_perp / perp_size.unsqueeze(-1)
            self.embedding.data.copy_(embed_normalized)
            self.expire_codes_(z_flattened)

            commit_loss = self.beta * (z_q.detach() - z).pow(2).mean() + (z_q - z.detach()).pow(2).mean()
            commit_loss += quant_loss
        else:
            commit_loss = torch.torch.tensor(0.0).to(z_q)

        # preserve gradients
        if self.rotation_trick:
            z_q = rotate_to(z, z_q)
        else:
            z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        if z.ndim == 4:
            z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        elif z.ndim == 3:
            z_q = rearrange(z_q, 'b h c -> b c h').contiguous()

        return (z_q, torch.tensor(0.0), min_encoding_indices), LossBreakdown(torch.tensor(0.0), torch.tensor(0.0),
                                                                             commit_loss, torch.tensor(0.0))

    def get_codebook_entry_and_token_with_feature(self, f, prompts=None, shape=None):
        assert f.shape[-1] == self.e_dim, "need to permute the dim into the last dimension!"
        shape = f.shape if shape is None else shape
        f_flattened = f.view(-1, self.e_dim)
        out = self._quantize(f_flattened, prompts, no_grad=True)
        out['z_q'] = out['z_q'].view(shape)
        return out

    def get_codebook_entry_with_token(self, token, prompts=None):
        assert token.shape[-1] == self.e_type_num + 1, f"token shape is not correct! the last dimension should be {self.e_type_num + 1}!"
        e = self.embedding.detach()
        prompts = e if prompts is None else prompts
        assert prompts.shape[-1] == self.e_dim
        norm_sem = self.get_norm_sem(reg=True, reg_term=prompts)

        sim_es = F.linear(e, norm_sem)  # K x P
        e_proj = sim_es.unsqueeze(-1) * norm_sem  # K x P x D
        e_perp = e - e_proj.sum(1)

        sim_zqs = torch.gather(sim_es.unsqueeze(0).repeat(token.shape[0], 1, 1), dim=1,
                               index=token[:-1].unsqueeze(1)).squeeze()  # L x P
        z_q_proj = sim_zqs.unsqueeze(-1) * norm_sem  # L x P x D
        z_q_perp = F.one_hot(token[-1], self.n_e).to(e_perp) @ e_perp
        z_q = torch.sum(z_q_proj, dim=1) + z_q_perp  # L x D
        return z_q

    @torch.no_grad()
    def get_codebook_size_based_semantic(self, reg=False):
        prompts = self.embedding.detach() if reg else None
        norm_sem = self.get_norm_sem(reg=reg, reg_term=prompts)
        sim_es = self.embedding @ norm_sem.t()  # K x P
        sizes = [torch.unique(sim_es[:, k]).numel() for k in range(self.e_type_num)]
        size = math.prod(sizes)
        return size, sizes

