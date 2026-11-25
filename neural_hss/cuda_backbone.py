"""Fused CUDA path for the Neural-HSS backbone (HSSLayer + TrainableLeakyReLU), any ndim.

The HSS factor of every CP term is assembled into a dense matrix (one kernel
launch, differentiable) and applied along every axis, and every mode product
of Eq. (1), forward and backward, runs as a strided GEMM on TF32 tensor cores
without permute copies (``csrc/hss_cuda.cu``):

* ``precision="fp32"``: 3xTF32 split, FP32-level accuracy (default),
* ``precision="tf32"``: single TF32 pass, faster, ~5e-4 relative error.

For a layer with m axes, CP rank K, input x: (Bt, I_0, ..., I_{m-1}) and dense
factors W[j]: (K, O_j, I_j) (all the same tensor for an HSSLayer), the forward
pass is

* axis m-1 for every term in one GEMM (terms stacked along N),
* axes m-2, ..., 1 in one GEMM each (terms and slices in the batch),
* axis 0 with the terms stacked along K, plus the bias (along axis m-2) and the
  LeakyReLU (none after the output layer) in the epilogue,

and m = 1 is a single GEMM. The backward pass runs the same contractions in
reverse order, plus one split-K GEMM per axis for the factor gradients.

The extension is compiled on first use with ``torch.utils.cpp_extension``,
which needs ``nvcc`` (matching the CUDA major version of PyTorch) and ``ninja``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.autograd.function import once_differentiable

_EXT = None
_BIG = 1 << 30  # block length meaning "not blocked"
TILE, KSTEP = 128, 32  # BM/BN and BK in hss_cuda.cu
ALIGN = 4  # spatial sizes must be multiples of this (16-byte vectors)
CHECK_BOUNDS = bool(os.environ.get("NEURAL_HSS_CHECK_BOUNDS"))


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        src = Path(__file__).resolve().parent / "csrc" / "hss_cuda.cu"
        _EXT = load(
            name="neural_hss_cuda",
            sources=[str(src)],
            extra_cuda_cflags=["-O3"],
            verbose=bool(os.environ.get("NEURAL_HSS_VERBOSE_BUILD")),
        )
    return _EXT


def is_supported(device: torch.device | None = None) -> bool:
    """True when CUDA is available and the GPU has TF32 tensor cores (sm_80+)."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device) >= (8, 0)


@lru_cache(maxsize=None)
def _num_sms(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _aligned(t: torch.Tensor) -> torch.Tensor:
    """Contiguous with a 16-byte aligned data pointer (needed by the 16-byte loads)."""
    t = t.contiguous()
    return t if t.data_ptr() % 16 == 0 else t.clone()


def _prod(sizes) -> int:
    out = 1
    for s in sizes:
        out *= s
    return out


# ---------------------------------------------------------------------------
# Generic strided GEMM:  C[z](m, n) = sum_k A[z](m, k) B[z](k, n)
# An index i of a blocked dimension maps to (i % blk) * stride + (i // blk) * block_stride.


def _op(sz=0, szb=None, s=0, sb=0, blk=_BIG, sk=0, skb=0):
    """Strides of A (index m) or B (index n): batch (sz, szb), M/N (s, sb, blk), K (sk, skb).

    The K block length is shared by A and B (``kblk`` of `gemm`); ``szb=None``
    means linear in the batch index (``szb = zblk * sz``).
    """
    return [sz, szb, s, sb, blk, sk, skb]


def _out(sz=0, szb=None, ssplit=0, sm=0, smb=0, mblk=_BIG, snb=0, nblk=_BIG):
    """Strides of C: batch (sz, szb), K split, M (sm, smb, mblk), N (unit stride, snb, nblk)."""
    return [sz, szb, ssplit, sm, smb, mblk, snb, nblk]


def _span(n, blk, s, sb):
    """Largest (i % blk) * s + (i // blk) * sb over 0 <= i < n (non-negative strides)."""
    q = (n - 1) // blk
    last = ((n - 1) % blk) * s + q * sb
    return max(last, (blk - 1) * s + (q - 1) * sb) if q > 0 else last


def _check_bounds(name, t, terms):
    end = sum(_span(*term) for term in terms)
    assert end < t.numel(), f"{name} out of bounds: offset {end} >= numel {t.numel()}"


def gemm(a, b, c, M, N, K, A, B, C, *, a_kc, b_kc, kblk=None, batch=1, zblk=_BIG, splits=1, tps=None,
         epi=0, c2=None, bias=None, slope=None, bias_on_n=False, bias_div=1, bias_mod=1, split3=True):
    """C[z](m, n) = sum_k A[z](m, k) B[z](k, n) on strided views; see `_op` and `_out`.

    ``a_kc`` / ``b_kc``: the operand is contiguous along K (else along M / N).
    ``epi=1`` adds ``bias`` (optional; indexed by m, or by ``(n // bias_div) %
    bias_mod`` when ``bias_on_n``), writes the result to ``c`` (optional) and
    LeakyReLU(``slope``) of it to ``c2`` (optional).
    """
    kblk = K if kblk is None else kblk
    if tps is None:
        tps = K // kblk * -(-kblk // KSTEP)
    for strides in (A, B, C):
        if strides[1] is None:
            strides[1] = zblk * strides[0] if zblk < _BIG else 0
    if CHECK_BOUNDS:
        zb = min(zblk, batch)
        for name, t, st, mn in (("A", a, A, M), ("B", b, B, N)):
            _check_bounds(name, t, [(batch, zb, st[0], st[1]), (mn, st[4], st[2], st[3]), (K, kblk, st[5], st[6])])
        for name, t in (("C", c), ("C2", c2)):
            if t is not None:
                _check_bounds(name, t, [(batch, zb, C[0], C[1]), (splits, 1, 0, C[2]), (M, C[5], C[3], C[4]),
                                        (N, C[7], 1, C[6])])
    _ext().gemm(a, b, c, c2, bias, slope, [M, N, K, kblk, batch, zblk, splits, tps], A, B, C,
                a_kc, b_kc, epi, bias_on_n, bias_div, bias_mod, split3)


def _wgrad(a, b, M, N, K, A, B, C, shape, *, kblk, a_kc, b_kc, batch=1, split3=True):
    """Weight-gradient GEMM with a long reduction: split-K into a workspace, then sum."""
    ktiles = K // kblk * -(-kblk // KSTEP)
    out_tiles = -(-M // TILE) * -(-N // TILE) * batch
    splits = max(1, min(2 * _num_sms(a.device.index) // out_tiles, ktiles))
    tps = -(-ktiles // splits)
    splits = -(-ktiles // tps)
    out = a.new_empty((splits, *shape) if splits > 1 else shape)
    C[2] = _prod(shape)
    gemm(a, b, out, M, N, K, A, B, C, a_kc=a_kc, b_kc=b_kc, kblk=kblk, batch=batch, splits=splits, tps=tps,
         split3=split3)
    return out.sum(0) if splits > 1 else out


def _gemm_split(a, b, shape, M, N, K, A, B, C, *, kblk=None, batch=1, **kw):
    """Split-K version of a GEMM whose output tiles alone would leave most SMs idle.

    Returns the output (a new tensor of `shape`, which C must address contiguously),
    or None when the GEMM should run unsplit: enough output tiles, or a reduction
    too short (< 32 K tiles) for the split to pay for the extra reduction kernels.
    """
    kblk = K if kblk is None else kblk
    ktiles = K // kblk * -(-kblk // KSTEP)
    items = -(-M // TILE) * -(-N // TILE) * batch
    sms = _num_sms(a.device.index)
    if 2 * items > sms or ktiles < 32:
        return None
    splits = min(sms // items, ktiles // 4)  # >= 4 K tiles per split
    if splits < 2:
        return None
    tps = -(-ktiles // splits)
    splits = -(-ktiles // tps)
    ws = a.new_empty((splits, *shape))
    C[2] = _prod(shape)
    gemm(a, b, ws, M, N, K, A, B, C, kblk=kblk, batch=batch, splits=splits, tps=tps, **kw)
    return ws.sum(0)


# ---------------------------------------------------------------------------
# Structured mode products (hss_kernel): the HSS factors applied block by block,
# from packed factors, instead of a GEMM with the dense factor. Opt-in: at the paper's
# size (128^3, 2 levels, rank 4) on an idle A100 they are no faster than the dense
# GEMMs with 3xTF32 (both are limited by memory traffic) and slower with TF32.

STRUCTURED = os.environ.get("NEURAL_HSS_STRUCTURED", "0") == "1"
# Structured factor gradients too (hss_wgrad_kernel, from the routed projections that
# the structured products save; implies STRUCTURED): ~10% faster per call than the dense
# GEMMs at 128^3, but saving the projections costs time and memory, and the layer is no faster.
STRUCTURED_WGRAD = os.environ.get("NEURAL_HSS_STRUCTURED_WGRAD", "0") == "1"
_HB_OT, _HB_PT = 16, 8  # hss_kernel: O, I <= 8 * _HB_OT (128), NL * R <= 8 * _HB_PT (64)


def _xs(sz=0, s=0, sb=0, blk=_BIG, sk=0, skb=0):
    """X(z, m, t I + i) of the structured kernel: z sz + (m % blk) s + (m // blk) sb + i sk + t skb."""
    return [sz, s, sb, blk, sk, skb]


def _cs(sz=0, sm=0, smb=0, mblk=_BIG, sn=0):
    """C(z, m, o) of the structured kernel: z sz + (m % mblk) sm + (m // mblk) smb + o sn."""
    return [sz, sm, smb, mblk, sn]


def _struct_dims(O, I, L, r):
    """Leaves, padded rank, leaf columns, pitches and floats per term of a packed factor."""
    NL = 1 << L
    R = -(-(L * r) // 8) * 8
    ci = I // NL
    dp, ep, pp = ci + 4, R + 4, NL * R + 4
    fstride = -(-(O * dp + NL * R * dp + O * ep) // 4) * 4
    return NL, R, ci, dp, ep, pp, fstride


class Struct:
    """Packed factors of a layer for the structured kernels: `fwd` applies W, `bwd` W^T.

    `idx_f` maps the forward pack to cat(params) + [0] (padding to the last slot),
    which scatters a packed factor gradient back to the parameters; `wgrad` says
    whether the structured factor-gradient kernel applies.
    """

    __slots__ = ("L", "r", "terms", "fwd", "bwd", "idx_f", "total", "wgrad")

    def __init__(self, L, r, terms, fwd, bwd, idx_f=None, total=0, wgrad=False):
        self.L, self.r, self.terms, self.fwd, self.bwd = L, r, terms, fwd, bwd
        self.idx_f, self.total, self.wgrad = idx_f, total, wgrad


def hss_mode(x, fact, c, M, O, I, S, X, C, *, sum_terms, last, epi=0, c2=None, bias=None, slope=None,
             bias_on_n=False, bias_div=1, bias_mod=1, split3=True, proj=None):
    """C[z](m, o) = sum_t sum_i X[z](m, t I + i) W_t(o, i) with the packed factors `fact`; see hss_kernel.

    `proj`, a (terms, M, NL R) tensor (`_proj`), receives the routed projections
    for the structured factor gradient; with `proj` and c = c2 = None, only
    those are computed.
    """
    NL, R = 1 << S.L, -(-(S.L * S.r) // 8) * 8
    _ext().hss_mode(x, fact, c, c2, bias, slope, proj, [M, O, I, S.terms, int(sum_terms), NL, S.L, S.r, R], X, C,
                    last, epi, bias_on_n, bias_div, bias_mod, split3)


def _proj(x, S, M):
    """An empty buffer for the routed projections of M fibers (see hss_mode)."""
    return x.new_empty(S.terms, M, (1 << S.L) * (-(-(S.L * S.r) // 8) * 8))


def _output_struct(x, fact, shape, M, O, I, S, X, C, *, sum_terms, last, bias, bias_kw, slope, save, split3,
                   proj=None):
    """`_output_gemm` for a structured product: bias and activation always in the kernel."""
    kw = dict(sum_terms=sum_terms, last=last, split3=split3, bias=bias, proj=proj, **bias_kw)
    if slope is None:
        Z = x.new_empty(shape)
        hss_mode(x, fact, Z, M, O, I, S, X, C, epi=0 if bias is None else 1, **kw)
        return Z, None
    Y = x.new_empty(shape) if save else None
    Z = x.new_empty(shape)
    hss_mode(x, fact, Y, M, O, I, S, X, C, epi=1, c2=Z, slope=slope, **kw)
    return Z, Y


# ---------------------------------------------------------------------------
# CP-HSS layer + LeakyReLU on dense factors.


def _sizes(x, W):
    Bt, I = x.shape[0], list(x.shape[1:])
    O = [w.shape[1] for w in W]
    Kt = W[0].shape[0]
    if len(I) != len(W) or any(tuple(w.shape) != (Kt, o, i) for w, o, i in zip(W, O, I)):
        raise ValueError(f"factor shapes {[tuple(w.shape) for w in W]} do not match the input {tuple(x.shape)}")
    return Bt, I, O, Kt


def _output_gemm(a, b, shape, M, N, K, A, B, C, *, bias, bias_kw, field, slope, save, **kw):
    """The last GEMM of a layer (output `shape`, contiguous), with the bias and activation.

    `bias_kw` places the bias in the epilogue, and `field` is the output's
    (Bt, O_0, ..., O_{m-1}) shape for the bias of the split-K path; `slope`
    None means no activation. Returns (Z, Y), Y the pre-activation when there
    is an activation and ``save``.
    """
    Y = _gemm_split(a, b, shape, M, N, K, A, B, C, **kw)
    if Y is not None:  # split-K: bias and activation after the reduction
        if bias is not None:
            Y.view(field).add_(bias.view(1, -1, 1))
        if slope is None:
            return Y, None
        return torch.where(Y >= 0, Y, Y * slope), (Y if save else None)
    if slope is None:
        Z = a.new_empty(shape)
        gemm(a, b, Z, M, N, K, A, B, C, epi=0 if bias is None else 1, **bias_kw, **kw)
        return Z, None
    Y = a.new_empty(shape) if save else None
    Z = a.new_empty(shape)
    gemm(a, b, Y, M, N, K, A, B, C, epi=1, c2=Z, slope=slope, **bias_kw, **kw)
    return Z, Y


def forward(x, W, bias, slope, split3=True, save=True, S=None, projs=None):
    """Z = act(sum_k x x_0 W[0][k] ... x_{m-1} W[m-1][k] + bias).

    x: (Bt, I_0, ..., I_{m-1}), contiguous; W[j]: (K, O_j, I_j), dense and
    contiguous; bias: (O_{m-2},), added along axis m-2 (``bias.view(1, -1, 1)``
    of the original code), or None (always for m = 1); slope: (1,), or None for
    no activation. With `S` (a `Struct`, the factors of every axis being the
    same), the products run on the structured kernel, and W only gives the sizes.
    Returns Z, Ts and Y for `backward`: Ts[j] is the input of the axis-j product
    (j < m-1) and Y the pre-activation (Ts empty when ``save=False``, Y None then
    or without activation). With `S` and `projs` (a list of m slots), projs[j]
    receives the routed projections of the axis-j product, for the structured
    factor gradients.
    """
    m = len(W)
    Bt, I, O, Kt = _sizes(x, W)
    kw = dict(split3=split3)

    def proj(j, M):
        if projs is None or S is None:
            return None
        projs[j] = _proj(x, S, M)
        return projs[j]

    if m == 1:
        if bias is not None:
            raise ValueError("a 1D layer has no bias")
        # Y[b, o] = sum_i x[b, i] W[0][0, o, i]
        if S is not None:
            Z, Y = _output_struct(x, S.fwd, (Bt, O[0]), Bt, O[0], I[0], S, _xs(s=I[0], sk=1), _cs(sm=O[0], sn=1),
                                  sum_terms=False, last=True, bias=None, bias_kw={}, slope=slope, save=save,
                                  proj=proj(0, Bt), **kw)
            return Z, [], Y
        A, B, C = _op(s=I[0], sk=1), _op(s=I[0], sk=1), _out(sm=O[0])
        Z, Y = _output_gemm(x, W[0], (Bt, O[0]), Bt, O[0], I[0], A, B, C, bias=None, bias_kw={}, field=None,
                            slope=slope, save=save, a_kc=True, b_kc=True, **kw)
        return Z, [], Y

    Ts = [None] * (m - 1)
    # axis m-1, every term in one GEMM: T[k, r, o] = sum_i x[r, i] W[m-1][k, o, i]
    R = Bt * _prod(I[:-1])
    T = x.new_empty(Kt, R, O[-1])
    if S is not None:
        hss_mode(x, S.fwd, T, R, O[-1], I[-1], S, _xs(s=I[-1], sk=1), _cs(sz=R * O[-1], sm=O[-1], sn=1),
                 sum_terms=False, last=True, proj=proj(m - 1, R), **kw)
    else:
        gemm(x, W[-1], T, R, Kt * O[-1], I[-1],
             _op(s=I[-1], sk=1), _op(s=I[-1], sk=1), _out(sm=O[-1], nblk=O[-1], snb=R * O[-1]),
             a_kc=True, b_kc=True, **kw)
    # axes m-2 .. 1, terms and slices in the batch (z = k * outer + s): T'[k, s] = W[j][k] T[k, s]
    for j in range(m - 2, 0, -1):
        Ts[j] = T if save else None
        outer, inner = Bt * _prod(I[:j]), _prod(O[j + 1:])
        Tn = x.new_empty(Kt, outer, O[j], inner)
        if S is not None:  # fibers (s, n), z = k
            hss_mode(T, S.fwd, Tn, outer * inner, O[j], I[j], S,
                     _xs(sz=outer * I[j] * inner, s=1, sb=I[j] * inner, blk=inner, sk=inner),
                     _cs(sz=outer * O[j] * inner, sm=1, smb=O[j] * inner, mblk=inner, sn=inner),
                     sum_terms=False, last=False, proj=proj(j, outer * inner), **kw)
        else:
            gemm(W[j], T, Tn, O[j], inner, I[j],
                 _op(szb=O[j] * I[j], s=I[j], sk=1), _op(sz=I[j] * inner, sk=inner, s=1),
                 _out(sz=O[j] * inner, sm=inner),
                 a_kc=True, b_kc=False, batch=Kt * outer, zblk=outer, **kw)
        T = Tn
    # axis 0, terms stacked along K, bias + LeakyReLU in the epilogue:
    # Y[b, o, n] = sum_{k, i} W[0][k, o, i] T[k, b, i, n] + bias
    Ts[0] = T if save else None
    inner = _prod(O[1:])
    if S is not None:  # fibers (b, n), terms summed in the kernel; the bias is along o_0 (m = 2) or in n
        bias_kw = dict(bias_on_n=True, bias_div=1, bias_mod=O[0]) if m == 2 else \
            dict(bias_on_n=False, bias_div=O[-1], bias_mod=O[-2])
        Z, Y = _output_struct(T, S.fwd, (Bt, O[0], inner), Bt * inner, O[0], I[0], S,
                              _xs(s=1, sb=I[0] * inner, blk=inner, sk=inner, skb=Bt * I[0] * inner),
                              _cs(sm=1, smb=O[0] * inner, mblk=inner, sn=inner),
                              sum_terms=True, last=False, bias=bias, bias_kw=bias_kw, slope=slope, save=save,
                              proj=proj(0, Bt * inner), **kw)
        return Z.view(Bt, *O), (Ts if save else []), Y
    A = _op(s=I[0], sk=1, skb=O[0] * I[0])
    B = _op(sz=I[0] * inner, sk=inner, skb=Bt * I[0] * inner, s=1)
    C = _out(sz=O[0] * inner, sm=inner)
    # the bias axis m-2 is the GEMM row for m = 2, else part of the column n = (o_1, ..., o_{m-1})
    bias_kw = {}
    if bias is not None:
        bias_kw = dict(bias=bias, bias_on_n=True, bias_div=O[-1], bias_mod=O[-2]) if m > 2 else dict(bias=bias)
    Z, Y = _output_gemm(W[0], T, (Bt, O[0], inner), O[0], inner, Kt * I[0], A, B, C, bias=bias, bias_kw=bias_kw,
                        field=(Bt, *O), slope=slope, save=save, a_kc=True, b_kc=False, kblk=I[0], batch=Bt, **kw)
    return Z.view(Bt, *O), (Ts if save else []), Y


def backward(dZ, x, W, slope, Ts, Y, has_bias, split3=True, need_dx=True, S=None, wpack=None, projs=None):
    """Gradients (dx or None, [dW_j], dbias or None, dslope or None) of `forward`.

    With `S`, the input gradients run on the structured kernel (transposed packs).
    With `wpack` too, a (terms, fstride) tensor, and the `projs` of `forward`, the
    factor gradients of every axis are added to it in the packed layout by the
    structured kernel (dW is then all None); otherwise they are dense GEMMs.
    """
    m = len(W)
    Bt, I, O, Kt = _sizes(x, W)
    kw = dict(split3=split3)
    dZ = _aligned(dZ)
    dbias = dslope = None
    if slope is not None:
        # dY = dZ * act'(Y), with sums along the last axis for the bias and slope gradients
        dY = torch.empty_like(Y)
        rows = Y.numel() // O[-1]
        row_db, row_ds = x.new_empty(rows), x.new_empty(rows)
        _ext().act_backward(dZ, Y, slope, dY, row_db, row_ds, O[-1])
        dslope = row_ds.sum()
        if has_bias:  # the bias varies along axis m-2
            dbias = row_db.view(-1, O[-2]).sum(0)
    else:
        dY = dZ
        if has_bias:
            dbias = dY.view(-1, O[-2], O[-1]).sum((0, 2))
    dW = [None] * m

    if m == 1:
        dX = None
        Q = None if wpack is None else _proj(x, S, Bt)  # routed projections of dY, for the factor gradient
        if S is not None and (need_dx or Q is not None):
            dX = torch.empty_like(x) if need_dx else None
            hss_mode(dY, S.bwd, dX, Bt, I[0], O[0], S, _xs(s=O[0], sk=1), _cs(sm=I[0], sn=1),
                     sum_terms=False, last=True, proj=Q, **kw)
        elif need_dx:  # dX[b, i] = sum_o dY[b, o] W[0][0, o, i]
            A, B, C = _op(s=O[0], sk=1), _op(sk=I[0], s=1), _out(sm=I[0])
            dX = _gemm_split(dY, W[0], tuple(x.shape), Bt, I[0], O[0], A, B, C, a_kc=True, b_kc=False, **kw)
            if dX is None:
                dX = torch.empty_like(x)
                gemm(dY, W[0], dX, Bt, I[0], O[0], A, B, C, a_kc=True, b_kc=False, **kw)
        # dW[o, i] = sum_b dY[b, o] x[b, i]
        if wpack is not None:
            _wgrad_struct(x, dY, projs[0], Q, S, Bt, O[0], I[0], _ws(s=I[0], sk=1), _ws(s=O[0], sk=1), wpack,
                          last=True, **kw)
            return dX, dW, dbias, dslope
        dW[0] = _wgrad(dY, x, O[0], I[0], Bt, _op(s=1, sk=O[0]), _op(sk=I[0], s=1), _out(sm=I[0]),
                       (O[0], I[0]), kblk=Bt, a_kc=False, b_kc=False, **kw).unsqueeze(0)
        return dX, dW, dbias, dslope

    # axis 0: dT[k, b, i, n] = sum_o W[0][k, o, i] dY[b, o, n]   (terms stacked along M)
    inner = _prod(O[1:])
    dT = x.new_empty(Kt, Bt, I[0], inner)
    Q = None if wpack is None else _proj(x, S, Bt * inner)
    if S is not None:  # fibers (b, n), z = k
        hss_mode(dY, S.bwd, dT, Bt * inner, I[0], O[0], S,
                 _xs(s=1, sb=O[0] * inner, blk=inner, sk=inner),
                 _cs(sz=Bt * I[0] * inner, sm=1, smb=I[0] * inner, mblk=inner, sn=inner),
                 sum_terms=False, last=False, proj=Q, **kw)
    else:
        gemm(W[0], dY, dT, Kt * I[0], inner, O[0],
             _op(s=1, blk=I[0], sb=O[0] * I[0], sk=I[0]), _op(sz=O[0] * inner, sk=inner, s=1),
             _out(sz=I[0] * inner, sm=inner, mblk=I[0], smb=Bt * I[0] * inner),
             a_kc=False, b_kc=False, batch=Bt, **kw)
    #         dW[0][k, o, i] = sum_{b, n} dY[b, o, n] Ts[0][k, b, i, n]   (terms stacked along N)
    if wpack is not None:  # X = Ts[0] per term, G = dY shared by the terms; fibers (b, n)
        _wgrad_struct(Ts[0], dY, projs[0], Q, S, Bt * inner, O[0], I[0],
                      _ws(sz=Bt * I[0] * inner, s=1, sb=I[0] * inner, blk=inner, sk=inner),
                      _ws(s=1, sb=O[0] * inner, blk=inner, sk=inner), wpack, last=False, **kw)
    else:
        dW[0] = _wgrad(dY, Ts[0], O[0], Kt * I[0], Bt * inner,
                       _op(s=inner, sk=1, skb=O[0] * inner),
                       _op(s=inner, blk=I[0], sb=Bt * I[0] * inner, sk=1, skb=I[0] * inner),
                       _out(sm=I[0], nblk=I[0], snb=O[0] * I[0]),
                       (Kt, O[0], I[0]), kblk=inner, a_kc=True, b_kc=True, **kw)
    # axes 1 .. m-2: dT'[k, s] = W[j][k]^T dT[k, s] and dW[j][k] = sum_s dT[k, s] Ts[j][k, s]^T
    for j in range(1, m - 1):
        outer, inner = Bt * _prod(I[:j]), _prod(O[j + 1:])
        dTn = x.new_empty(Kt, outer, I[j], inner)
        Q = None if wpack is None else _proj(x, S, outer * inner)
        if S is not None:
            hss_mode(dT, S.bwd, dTn, outer * inner, I[j], O[j], S,
                     _xs(sz=outer * O[j] * inner, s=1, sb=O[j] * inner, blk=inner, sk=inner),
                     _cs(sz=outer * I[j] * inner, sm=1, smb=I[j] * inner, mblk=inner, sn=inner),
                     sum_terms=False, last=False, proj=Q, **kw)
        else:
            gemm(W[j], dT, dTn, I[j], inner, O[j],
                 _op(szb=O[j] * I[j], s=1, sk=I[j]), _op(sz=O[j] * inner, sk=inner, s=1),
                 _out(sz=I[j] * inner, sm=inner),
                 a_kc=False, b_kc=False, batch=Kt * outer, zblk=outer, **kw)
        if wpack is not None:  # X = Ts[j], G = dT, both per term; fibers (s, n)
            _wgrad_struct(Ts[j], dT, projs[j], Q, S, outer * inner, O[j], I[j],
                          _ws(sz=outer * I[j] * inner, s=1, sb=I[j] * inner, blk=inner, sk=inner),
                          _ws(sz=outer * O[j] * inner, s=1, sb=O[j] * inner, blk=inner, sk=inner),
                          wpack, last=False, **kw)
        else:
            dW[j] = _wgrad(dT, Ts[j], O[j], I[j], outer * inner,
                           _op(sz=outer * O[j] * inner, s=inner, sk=1, skb=O[j] * inner),
                           _op(sz=outer * I[j] * inner, s=inner, sk=1, skb=I[j] * inner),
                           _out(sz=O[j] * I[j], sm=I[j]),
                           (Kt, O[j], I[j]), kblk=inner, batch=Kt, a_kc=True, b_kc=True, **kw)
        dT = dTn
    # axis m-1: dX[r, i] = sum_{k, o} dT[k, r, o] W[m-1][k, o, i]   (terms stacked along K)
    #           dW[m-1][k, o, i] = sum_r dT[k, r, o] x[r, i]          (terms stacked along M)
    R, Ol, Il = Bt * _prod(I[:-1]), O[-1], I[-1]
    dX = None
    Q = None if wpack is None else _proj(x, S, R)
    if S is not None and (need_dx or Q is not None):  # terms summed in the kernel
        dX = torch.empty_like(x) if need_dx else None
        hss_mode(dT, S.bwd, dX, R, Il, Ol, S, _xs(s=Ol, sk=1, skb=R * Ol), _cs(sm=Il, sn=1),
                 sum_terms=True, last=True, proj=Q, **kw)
    elif need_dx:
        A, B, C = _op(s=Ol, sk=1, skb=R * Ol), _op(sk=Il, skb=Ol * Il, s=1), _out(sm=Il)
        dX = _gemm_split(dT, W[-1], tuple(x.shape), R, Il, Kt * Ol, A, B, C, a_kc=True, b_kc=False, kblk=Ol, **kw)
        if dX is None:
            dX = torch.empty_like(x)
            gemm(dT, W[-1], dX, R, Il, Kt * Ol, A, B, C, a_kc=True, b_kc=False, kblk=Ol, **kw)
    if wpack is not None:  # X = x shared by the terms, G = dT per term
        _wgrad_struct(x, dT, projs[m - 1], Q, S, R, Ol, Il, _ws(s=Il, sk=1), _ws(sz=R * Ol, s=Ol, sk=1), wpack,
                      last=True, **kw)
    else:
        dW[-1] = _wgrad(dT, x, Kt * Ol, Il, R,
                        _op(s=1, blk=Ol, sb=R * Ol, sk=Ol), _op(sk=Il, s=1), _out(sm=Il),
                        (Kt, Ol, Il), kblk=R, a_kc=False, b_kc=False, **kw)
    return dX, dW, dbias, dslope


class HSSFunction(torch.autograd.Function):
    """Autograd wrapper of `forward` / `backward`; W are the dense factors.

    First-order only: the backward runs CUDA kernels autograd cannot see, so a
    second derivative (``create_graph=True``) raises instead of silently missing
    the HSS terms. Use ``backend="torch"`` for those.
    """

    @staticmethod
    def forward(ctx, x, bias, slope, split3, S, *W):
        x = _aligned(x)
        slope1 = None if slope is None else slope.reshape(1)
        Z, Ts, Y = forward(x, W, None if bias is None else bias.contiguous(), slope1, split3=split3, save=True, S=S)
        ctx.split3, ctx.m, ctx.has_bias, ctx.S = split3, len(W), bias is not None, S
        ctx.save_for_backward(x, slope1, Y, *W, *Ts)
        return Z

    @staticmethod
    @once_differentiable
    def backward(ctx, dZ):
        x, slope1, Y, *rest = ctx.saved_tensors
        W, Ts = rest[: ctx.m], list(rest[ctx.m:])
        dX, dW, dbias, dslope = backward(
            dZ, x, W, slope1, Ts, Y, ctx.has_bias, split3=ctx.split3, need_dx=ctx.needs_input_grad[0], S=ctx.S
        )
        return (dX, dbias, None if dslope is None else dslope.reshape(()), None, None, *dW)


class HSSStructFunction(torch.autograd.Function):
    """`HSSFunction` with every product and the factor gradients on the structured
    kernels: the HSS parameters are the inputs, and no dense factor is built.

    First-order only, as `HSSFunction`.
    """

    @staticmethod
    def forward(ctx, x, bias, slope, split3, S, shape, *params):
        x = _aligned(x)
        slope1 = None if slope is None else slope.reshape(1)
        m = len(x.shape[1:])
        W = (x.new_empty(1).expand(shape),) * m  # sizes only
        projs = [None] * m  # routed projections of every product, for the factor gradients
        Z, Ts, Y = forward(x, W, None if bias is None else bias.contiguous(), slope1, split3=split3, save=True, S=S,
                           projs=projs)
        ctx.split3, ctx.has_bias, ctx.S, ctx.shape, ctx.nts = split3, bias is not None, S, shape, len(Ts)
        ctx.numels = [p.numel() for p in params]
        ctx.shapes = [p.shape for p in params]
        ctx.save_for_backward(x, slope1, Y, *Ts, *projs)
        return Z

    @staticmethod
    @once_differentiable
    def backward(ctx, dZ):
        x, slope1, Y, *rest = ctx.saved_tensors
        Ts, projs = list(rest[: ctx.nts]), list(rest[ctx.nts:])
        S = ctx.S
        W = (x.new_empty(1).expand(ctx.shape),) * len(x.shape[1:])
        wpack = x.new_zeros(S.terms, S.fwd.numel() // S.terms)
        dX, _, dbias, dslope = backward(dZ, x, W, slope1, Ts, Y, ctx.has_bias, split3=ctx.split3,
                                        need_dx=ctx.needs_input_grad[0], S=S, wpack=wpack, projs=projs)
        flat = wpack.new_zeros(S.total + 1).index_add_(0, S.idx_f, wpack.reshape(-1))
        grads, off = [], 0
        for n, shp in zip(ctx.numels, ctx.shapes):
            grads.append(flat[off:off + n].view(shp))
            off += n
        return (dX, dbias, None if dslope is None else dslope.reshape(()), None, None, None, *grads)


# ---------------------------------------------------------------------------
# Dense assembly of the HSS factors of a layer, one kernel launch each way.


def _nodes(layer):
    """The HSS factor (an HSSNode tree) of every CP term of an HSSLayer, shared by all axes."""
    if layer.ndim == 1:
        return [layer.root]
    nodes = [factors[0] for factors in layer.hss]
    if any(f is not node for node, factors in zip(nodes, layer.hss) for f in factors):
        raise ValueError("The CUDA backbone expects one HSS factor per CP term, shared by every axis.")
    return nodes


def _first_param(layer):
    """The first parameter of the plan's walk: detects replaced parameters cheaply."""
    node = layer.root if layer.ndim == 1 else layer.hss[0][0]
    while not node.is_leaf:
        node = node.child1
    return node.full


def _layer_plan(layer):
    """HSS parameters, block table and dense shape of `layer` (cached on it)."""
    key = _first_param(layer)
    plan = getattr(layer, "_cuda_plan", None)
    if plan is not None and plan[3] is key:
        return plan
    params, rows = [], []

    def pid(t):
        params.append(t)
        return len(params) - 1

    nodes = _nodes(layer)
    for k, root in enumerate(nodes):

        def walk(node, r0, c0):
            if node.is_leaf:
                rows.append([pid(node.full), -1, 0, k, r0, c0, node.rows, node.cols])
                return
            walk(node.child1, r0, c0)
            walk(node.child2, r0 + node.r1, c0 + node.c1)
            rows.append([pid(node.U12), pid(node.V12), 0, k, r0, c0 + node.c1, node.r1, node.c2])
            rows.append([pid(node.U21), pid(node.V21), 0, k, r0 + node.r1, c0, node.r2, node.c1])

        walk(root, 0, 0)
    if len({id(p) for p in params}) != len(params):
        raise ValueError("The CUDA backbone does not support HSS parameters shared between CP terms.")
    meta = torch.tensor(rows, dtype=torch.int64)
    shapes = [(len(nodes), layer.out_sizes[0], layer.in_sizes[0])]
    plan = (params, meta, shapes, key)
    layer._cuda_plan = plan
    return plan


def _tree(root, L):
    """Leaves (left to right) and the interior nodes of each level of an HSSNode tree."""
    leaves, levels = [], [[] for _ in range(L)]

    def walk(node, depth):
        if node.is_leaf:
            leaves.append(node)
            return
        levels[depth].append(node)
        walk(node.child1, depth + 1)
        walk(node.child2, depth + 1)

    walk(root, 0)
    return leaves, levels


def _pack_index(nodes, L, r, O, I, offsets, zero, transposed):
    """Flat indices (into cat(params) + [0]) of the packed factors of W, or of W^T.

    See hss_kernel for the layout. For segment j at level l, the U rows of output
    segment j come from U12 (first half of its node) or U21, and the V rows of
    input segment j from V12 (second half of its node) or V21. W^T swaps the roles:
    leaves transposed, projections on the U rows, expansions by the V rows.
    """
    NL = 1 << L
    ro_w, ci_w = O // NL, I // NL  # leaf rows and columns of W
    Oo, Ii = (I, O) if transposed else (O, I)
    _, R, ci, dp, ep, _, fstride = _struct_dims(Oo, Ii, L, r)
    ro = Oo // NL
    idx = np.full(len(nodes) * fstride, zero, dtype=np.int64)
    a_ro, b_ci = np.arange(ro)[:, None], np.arange(ci)[None, :]
    t_col, t_row = np.arange(r)[None, :], np.arange(r)[:, None]
    for k, root in enumerate(nodes):
        leaves, levels = _tree(root, L)
        D0 = k * fstride
        P0 = D0 + Oo * dp
        E0 = P0 + NL * R * dp
        for j, leaf in enumerate(leaves):  # D[j ro + a][b] = leaf_j[a][b], or leaf_j[b][a] for W^T
            src = b_ci * ci_w + a_ro if transposed else a_ro * ci_w + b_ci
            idx[D0 + (j * ro + a_ro) * dp + b_ci] = offsets[id(leaf.full)] + src
        for l in range(L):
            sh = L - l - 1
            for j in range(NL):
                q = j >> (L - l)
                first, h = q << (L - l), (j >> sh) & 1
                node = levels[l][q]
                U, useg = (node.U21, j - first - (1 << sh)) if h else (node.U12, j - first)
                V, vseg = (node.V12, j - first - (1 << sh)) if h else (node.V21, j - first)
                (Pf, pseg, plen), (Ef, eseg, elen) = ((U, useg, ro_w), (V, vseg, ci_w)) if transposed else \
                    ((V, vseg, ci_w), (U, useg, ro_w))
                # P[j R + l r + t][b] = Pf[pseg plen + b][t];  E[j ro + a][l r + t] = Ef[eseg elen + a][t]
                idx[P0 + (j * R + l * r + t_row) * dp + b_ci] = offsets[id(Pf)] + (pseg * plen + b_ci) * r + t_row
                idx[E0 + (j * ro + a_ro) * ep + l * r + t_col] = offsets[id(Ef)] + (eseg * elen + a_ro) * r + t_col
    return idx


def struct_reason(layer, device=None) -> str | None:
    """Why the structured kernel cannot run `layer` (then dense GEMMs do), or None."""
    root = _nodes(layer)[0]
    L, r = root.levels, root.rank
    O, I = layer.out_sizes[0], layer.in_sizes[0]
    if not 1 <= L <= 3:
        return f"needs 1 to 3 HSS levels, got {L}"
    NL = 1 << L
    if max(O, I) > 8 * _HB_OT or O % (8 * NL) or I % (8 * NL):
        return f"needs sizes <= {8 * _HB_OT} with leaves of a multiple of 8, got {O} x {I} with {NL} leaves"
    _, R, *_ = _struct_dims(O, I, L, r)
    if NL * R > 8 * _HB_PT:
        return f"levels x rank too large ({L} x {r})"
    optin = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    terms = len(_nodes(layer))
    for Oo, Ii in ((O, I), (I, O)):
        _, _, _, _, _, pp, fstride = _struct_dims(Oo, Ii, L, r)
        if max(_ext().hss_smem(last, Ii, terms, fstride, pp) for last in (True, False)) > optin:
            return "the packed factors do not fit in shared memory"
    return None


def wgrad_reason(layer, device=None) -> str | None:
    """Why the structured factor-gradient kernel cannot run `layer` (dense GEMMs do), or None."""
    reason = struct_reason(layer, device)
    if reason is not None:
        return reason
    root = _nodes(layer)[0]
    L, r = root.levels, root.rank
    O, I = layer.out_sizes[0], layer.in_sizes[0]
    NL = 1 << L
    if O % (16 * NL) or I % (16 * NL):
        return f"needs leaves of a multiple of 16, got {O} x {I} with {NL} leaves"
    _, R, *_ = _struct_dims(O, I, L, r)
    optin = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    if max(_ext().hss_wgrad_smem(last, O, I, NL, R) for last in (True, False)) > optin:
        return "the factor-gradient tiles do not fit in shared memory"
    return None


def _struct(layer, params, device):
    """The packed factors (a `Struct`) of `layer`, or None when the structured kernel does not apply."""
    plan = getattr(layer, "_cuda_struct", None)
    if plan is None or plan[0] is not params[0] or plan[1] != device:
        info = None
        if struct_reason(layer, device) is None:
            nodes = _nodes(layer)
            L, r = nodes[0].levels, nodes[0].rank
            offsets, total = {}, 0
            for p in params:
                offsets[id(p)] = total
                total += p.numel()
            O, I = layer.out_sizes[0], layer.in_sizes[0]
            idx = [torch.from_numpy(_pack_index(nodes, L, r, O, I, offsets, total, t)).to(device) for t in (False, True)]
            info = (L, r, len(nodes), *idx, total, wgrad_reason(layer, device) is None)
        plan = (params[0], device, info)
        layer._cuda_struct = plan
    if plan[2] is None:
        return None
    L, r, terms, idx_f, idx_b, total, wgrad = plan[2]
    with torch.no_grad():
        flat = torch.cat([p.detach().reshape(-1) for p in params] + [params[0].new_zeros(1)])
        return Struct(L, r, terms, flat[idx_f], flat[idx_b], idx_f, total, wgrad)


def _wgrad_struct(X, G, P, Q, S, M, O, I, sx, sg, acc, *, last, split3):
    """Adds the packed factor gradient (terms, fstride) of one mode product to `acc`; see hss_wgrad_kernel.

    P, Q: the routed projections that the product and its input gradient saved (`hss_mode`).
    """
    NL, R = 1 << S.L, -(-(S.L * S.r) // 8) * 8
    cpt = max(1, _num_sms(X.device.index) // S.terms)
    ws = X.new_zeros(S.terms, cpt, acc.shape[1])
    _ext().hss_wgrad(X, G, P, Q, ws, [M, O, I, S.terms, NL, R], sx, sg, last, split3)
    acc += ws.sum(1)


def _ws(sz=0, s=0, sb=0, blk=_BIG, sk=0):
    """X(t, m, i) or G(t, m, o) of the factor-gradient kernel: t sz + (m % blk) s + (m // blk) sb + i sk."""
    return [sz, s, sb, blk, sk]


class HSSAssemble(torch.autograd.Function):
    """Dense factors, one (K, out, in) tensor per entry of `shapes`, from the HSS parameters."""

    @staticmethod
    def forward(ctx, meta, shapes, *params):
        dense = [params[0].new_empty(s) for s in shapes]
        _ext().assemble(list(params), dense, meta, False)
        ctx.meta = meta
        ctx.save_for_backward(*params)
        return tuple(dense)

    @staticmethod
    @once_differentiable
    def backward(ctx, *grads):
        params = ctx.saved_tensors
        pgrads = _ext().assemble(list(params), [g.contiguous() for g in grads], ctx.meta, True)
        return (None, None, *pgrads)


def unsupported_reason(layer) -> str | None:
    """Why `layer` (an HSSLayer) cannot use the CUDA path, or None if it can."""
    sizes = tuple(layer.in_sizes) + tuple(layer.out_sizes)
    if any(s % ALIGN for s in sizes):
        return f"the CUDA backbone needs every spatial size to be a multiple of {ALIGN}, got {sizes}"
    return None


def layer_forward(x, layer, act, precision: str = "fp32"):
    """Fused ``act(layer(x))`` for an HSSLayer (any ndim) and a TrainableLeakyReLU, or
    ``layer(x)`` when ``act`` is None (the output layer)."""
    if precision not in ("fp32", "tf32"):
        raise ValueError(f"precision must be 'fp32' or 'tf32', got {precision!r}")
    if x.dim() != layer.ndim + 1 or tuple(x.shape[1:]) != tuple(layer.in_sizes):
        raise ValueError(f"Expected input of shape (batch, {', '.join(map(str, layer.in_sizes))}), "
                         f"got {tuple(x.shape)}.")
    if x.shape[0] == 0:
        return x.new_empty((0, *layer.out_sizes))
    split3 = precision == "fp32"
    params, meta, shapes, _ = _layer_plan(layer)
    S = _struct(layer, params, x.device) if STRUCTURED or STRUCTURED_WGRAD else None
    bias = layer.bias
    slope = None if act is None else act.slope
    needs_grad = torch.is_grad_enabled() and (
        x.requires_grad
        or any(t is not None and t.requires_grad for t in (bias, slope))
        or any(p.requires_grad for p in params)
    )
    if needs_grad:
        if S is not None and S.wgrad and STRUCTURED_WGRAD:  # structured factor gradients, to the HSS parameters
            return HSSStructFunction.apply(x, bias, slope, split3, S, tuple(shapes[0]), *params)
        # the dense factor carries the factor gradients back to the HSS parameters
        (dense,) = HSSAssemble.apply(meta, shapes, *params)
        return HSSFunction.apply(x, bias, slope, split3, S, *(dense,) * layer.ndim)
    with torch.no_grad():  # inference: skip the saved intermediates (and the dense factor if structured)
        if S is not None:
            dense = x.new_empty(shapes[0])  # only its shape is used
        else:
            (dense,) = HSSAssemble.apply(meta, shapes, *params)
        Z, _, _ = forward(_aligned(x), (dense,) * layer.ndim, None if bias is None else bias.contiguous(),
                          None if slope is None else slope.reshape(1), split3=split3, save=False, S=S)
    return Z
