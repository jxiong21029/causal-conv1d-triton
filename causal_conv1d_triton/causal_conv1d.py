"""Triton causal depthwise conv1d over channels-last inputs, with forward and backward.

    out[b, t, d] = act(bias[d] + sum_k weight[d, k] * x[b, t - (W - 1) + k, d])

where positions t < 0 read `initial_states` (or zero). Follows the `causal_conv1d_fn` interface of
Dao-AILab's causal-conv1d package, except that x is (batch, seqlen, dim) and the states are
(batch, width - 1, dim): the states are literally the inputs preceding x[:, 0]. Supports any width
>= 2 and fp32/fp16/bf16 inputs (accumulation is fp32).
"""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _load_seq(seq_ptr, t, T, HAS_INIT: tl.constexpr):
    """Sequence id at row t. The initial states belong to the first sequence; all other rows
    outside [0, T) get -1."""
    if HAS_INIT:
        seq = tl.load(seq_ptr + tl.maximum(t, 0), mask=t < T, other=-1)
    else:
        seq = tl.load(seq_ptr + t, mask=(t >= 0) & (t < T), other=-1)
    return seq


@triton.jit
def _load_row(
    x_ptr, init_ptr, t, offs_d, mask_d, T, D, s_xt, HAS_INIT: tl.constexpr, PAD: tl.constexpr
):
    """Row t of concat(initial_states, x) in fp32, where rows in [-PAD, 0) are the initial states
    (a contiguous (PAD, D) block) and all other rows outside [0, T) are zero."""
    val = tl.load(x_ptr + t * s_xt + offs_d, mask=mask_d & (t >= 0) & (t < T), other=0.0).to(
        tl.float32
    )
    if HAS_INIT:
        val += tl.load(
            init_ptr + (t + PAD) * D + offs_d, mask=mask_d & (t < 0) & (t >= -PAD), other=0.0
        ).to(tl.float32)
    return val


@triton.jit
def _fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    init_ptr,
    seq_ptr,
    out_ptr,
    final_ptr,
    T,
    D,
    CHUNK,
    s_xb,
    s_xt,
    WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_INIT: tl.constexpr,
    HAS_SEQ: tl.constexpr,
    HAS_FINAL: tl.constexpr,
    SILU: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Each program walks forwards in time over one chunk of rows for one block of channels,
    keeping the previous PAD inputs in registers so each row of x is loaded once (plus PAD halo
    rows per chunk). Steps are unrolled in batches of ROWS so that the window shifts are free
    renames and each batch's loads are issued together."""
    PAD: tl.constexpr = WIDTH - 1
    pid_c = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_b = tl.program_id(2).to(tl.int64)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    x_ptr += pid_b * s_xb
    out_ptr += pid_b * T * D
    if HAS_INIT:
        init_ptr += pid_b * PAD * D
    if HAS_SEQ:
        seq_ptr += pid_b * T

    t_lo = pid_c * CHUNK
    t_hi = tl.minimum(t_lo + CHUNK, T)

    w = ()
    for k in tl.static_range(WIDTH):
        w = w + (tl.load(w_ptr + offs_d * WIDTH + k, mask=mask_d, other=0.0).to(tl.float32),)
    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    # Before the step at row t: xs = x[t - PAD : t], seqs = seq_idx[t - PAD : t].
    xs = ()
    for k in tl.static_range(PAD):
        xs = xs + (
            _load_row(x_ptr, init_ptr, t_lo - PAD + k, offs_d, mask_d, T, D, s_xt, HAS_INIT, PAD),
        )
    if HAS_SEQ:
        seqs = ()
        for k in tl.static_range(PAD):
            seqs = seqs + (_load_seq(seq_ptr, t_lo - PAD + k, T, HAS_INIT),)

    for i in range(tl.cdiv(t_hi - t_lo, ROWS)):
        t0 = t_lo + i * ROWS
        xb = ()
        for r in tl.static_range(ROWS):
            xb = xb + (
                _load_row(
                    x_ptr,
                    init_ptr,
                    t0 + r,
                    offs_d,
                    mask_d & (t0 + r < t_hi),
                    T,
                    D,
                    s_xt,
                    HAS_INIT,
                    PAD,
                ),
            )
        for r in tl.static_range(ROWS):
            t = t0 + r
            win = xs + (xb[r],)  # x[t - PAD : t + 1]
            if HAS_SEQ:
                sw = seqs + (_load_seq(seq_ptr, t, T, HAS_INIT),)  # seq_idx[t - PAD : t + 1]
                seq_t = sw[PAD]
            z = w[PAD] * win[PAD]
            for k in tl.static_range(PAD):
                if HAS_SEQ:
                    z += tl.where(sw[k] == seq_t, w[k] * win[k], 0.0)
                else:
                    z += w[k] * win[k]
            if HAS_BIAS:
                z += bias
            if HAS_SEQ:
                z = tl.where(seq_t >= 0, z, 0.0)
            if SILU:
                z = z * tl.sigmoid(z)
            tl.store(
                out_ptr + t * D + offs_d, z.to(out_ptr.dtype.element_ty), mask=mask_d & (t < t_hi)
            )

            xs = ()
            for k in tl.static_range(PAD):
                xs = xs + (win[k + 1],)
            if HAS_SEQ:
                seqs = ()
                for k in tl.static_range(PAD):
                    seqs = seqs + (sw[k + 1],)

    if HAS_FINAL:
        if t_hi == T:
            # Last PAD rows of concat(initial_states, x).
            for p in tl.static_range(PAD):
                row = _load_row(
                    x_ptr, init_ptr, T - PAD + p, offs_d, mask_d, T, D, s_xt, HAS_INIT, PAD
                )
                tl.store(
                    final_ptr + (pid_b * PAD + p) * D + offs_d,
                    row.to(final_ptr.dtype.element_ty),
                    mask=mask_d,
                )


@triton.jit
def _bwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    init_ptr,
    seq_ptr,
    dy_ptr,
    dfinal_ptr,
    dx_ptr,
    part_ptr,
    dinit_ptr,
    T,
    D,
    CHUNK,
    s_xb,
    s_xt,
    s_yb,
    s_yt,
    WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_INIT: tl.constexpr,
    HAS_SEQ: tl.constexpr,
    HAS_DFINAL: tl.constexpr,
    NEED_DINIT: tl.constexpr,
    SILU: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Each program walks backwards in time over one chunk of rows for one block of channels,
    keeping the last WIDTH inputs and the next PAD pre-activation gradients in registers, so each
    row of x and dy is loaded once. Chunks overlap by PAD halo rows. The first chunk continues
    into negative time to produce the initial-state gradients. Steps are unrolled in batches of
    ROWS so that the window shifts are free renames and each batch's loads are issued together."""
    PAD: tl.constexpr = WIDTH - 1
    pid_c = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_b = tl.program_id(2).to(tl.int64)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    x_ptr += pid_b * s_xb
    dy_ptr += pid_b * s_yb
    dx_ptr += pid_b * T * D
    if HAS_INIT:
        init_ptr += pid_b * PAD * D
    if HAS_DFINAL:
        dfinal_ptr += pid_b * PAD * D
    if NEED_DINIT:
        dinit_ptr += pid_b * PAD * D
    if HAS_SEQ:
        seq_ptr += pid_b * T

    t_lo = pid_c * CHUNK
    t_hi = tl.minimum(t_lo + CHUNK, T)
    t_first = tl.minimum(t_hi + PAD, T) - 1
    t_last = t_lo
    if NEED_DINIT:
        t_last = tl.where(pid_c == 0, -PAD, t_lo)

    w = ()
    for k in tl.static_range(WIDTH):
        w = w + (tl.load(w_ptr + offs_d * WIDTH + k, mask=mask_d, other=0.0).to(tl.float32),)
    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_d, mask=mask_d, other=0.0).to(tl.float32)

    # Before the step at row t: xs = x[t - PAD + 1 : t + 1], dzs = dz[t + 1 : t + PAD + 1],
    # seqs = seq_idx[t - PAD + 1 : t + PAD + 1].
    xs = ()
    dzs = ()
    for k in tl.static_range(PAD):
        xs = xs + (
            _load_row(
                x_ptr, init_ptr, t_first - PAD + 1 + k, offs_d, mask_d, T, D, s_xt, HAS_INIT, PAD
            ),
        )
        dzs = dzs + (tl.zeros([BLOCK_D], dtype=tl.float32),)
    if HAS_SEQ:
        seqs = ()
        for m in tl.static_range(2 * PAD):
            seqs = seqs + (_load_seq(seq_ptr, t_first - PAD + 1 + m, T, HAS_INIT),)
    dw = ()
    for k in tl.static_range(WIDTH):
        dw = dw + (tl.zeros([BLOCK_D], dtype=tl.float32),)
    db = tl.zeros([BLOCK_D], dtype=tl.float32)

    for i in range(tl.cdiv(t_first - t_last + 1, ROWS)):
        # Issue the loads for a batch of ROWS steps up front; the compiler cannot hoist loads
        # above the previous step's (possibly aliasing) stores.
        t0 = t_first - i * ROWS
        xb = ()
        dyb = ()
        for r in tl.static_range(ROWS):
            t = t0 - r
            valid = mask_d & (t >= t_last)
            xb = xb + (
                _load_row(x_ptr, init_ptr, t - PAD, offs_d, valid, T, D, s_xt, HAS_INIT, PAD),
            )
            dyb = dyb + (_load_row(dy_ptr, dy_ptr, t, offs_d, valid, T, D, s_yt, False, PAD),)

        for r in tl.static_range(ROWS):
            t = t0 - r
            win = (xb[r],) + xs  # x[t - PAD : t + 1]
            if HAS_SEQ:
                sw = (_load_seq(seq_ptr, t - PAD, T, HAS_INIT),) + seqs  # seq_idx[t-PAD : t+PAD+1]
                seq_t = sw[PAD]

            dz = dyb[r]
            if SILU:
                # Recompute the pre-activation in the same order as the forward.
                z = w[PAD] * win[PAD]
                for k in tl.static_range(PAD):
                    if HAS_SEQ:
                        z += tl.where(sw[k] == seq_t, w[k] * win[k], 0.0)
                    else:
                        z += w[k] * win[k]
                if HAS_BIAS:
                    z += bias
                sig = tl.sigmoid(z)
                dz = dz * sig * (1.0 + z * (1.0 - sig))
            if HAS_SEQ:
                dz = tl.where(seq_t >= 0, dz, 0.0)

            # x[t] feeds out[t + j] through weight[PAD - j].
            dx = w[PAD] * dz
            for j in tl.static_range(1, WIDTH):
                if HAS_SEQ:
                    dx += tl.where(sw[PAD + j] == seq_t, w[PAD - j] * dzs[j - 1], 0.0)
                else:
                    dx += w[PAD - j] * dzs[j - 1]
            if HAS_DFINAL:
                # final_states[p] = concat(initial_states, x)[T - PAD + p]
                f = t - (T - PAD)
                dx += tl.load(
                    dfinal_ptr + f * D + offs_d, mask=mask_d & (f >= 0) & (f < PAD), other=0.0
                ).to(tl.float32)

            own = (t >= t_lo) & (t < t_hi)
            tl.store(dx_ptr + t * D + offs_d, dx.to(dx_ptr.dtype.element_ty), mask=mask_d & own)
            if NEED_DINIT:
                tl.store(
                    dinit_ptr + (t + PAD) * D + offs_d,
                    dx.to(dinit_ptr.dtype.element_ty),
                    mask=mask_d & (t < 0) & (t >= t_last),
                )

            dz_own = tl.where(own, dz, 0.0)
            new_dw = ()
            for k in tl.static_range(WIDTH):
                if HAS_SEQ and k < PAD:
                    new_dw = new_dw + (dw[k] + tl.where(sw[k] == seq_t, dz_own * win[k], 0.0),)
                else:
                    new_dw = new_dw + (dw[k] + dz_own * win[k],)
            dw = new_dw
            db += dz_own

            xs = ()
            for k in tl.static_range(PAD):
                xs = xs + (win[k],)
            new_dzs = (dz,)
            for j in tl.static_range(PAD - 1):
                new_dzs = new_dzs + (dzs[j],)
            dzs = new_dzs
            if HAS_SEQ:
                seqs = ()
                for m in tl.static_range(2 * PAD):
                    seqs = seqs + (sw[m],)

    # Per-program partial sums, reduced over (batch, chunk) by _reduce_partials_kernel for
    # determinism. Row k < WIDTH holds dweight[:, k], row WIDTH holds dbias.
    part_ptr += (pid_b * tl.num_programs(0) + pid_c) * (WIDTH + 1) * D
    for k in tl.static_range(WIDTH):
        tl.store(part_ptr + k * D + offs_d, dw[k], mask=mask_d)
    if HAS_BIAS:
        tl.store(part_ptr + WIDTH * D + offs_d, db, mask=mask_d)


@triton.jit
def _reduce_partials_kernel(
    part_ptr,
    dw_ptr,
    db_ptr,
    P,
    D,
    N,
    WIDTH: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Sums the (P, WIDTH + 1, D) backward partials over P, writing columns n = k * D + d to
    dweight[d, k] for k < WIDTH and to dbias[d] for k == WIDTH. N is the number of columns used."""
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    acc = tl.zeros([BLOCK_P, BLOCK_N], dtype=tl.float32)
    for p0 in range(0, P, BLOCK_P):
        offs_p = p0 + tl.arange(0, BLOCK_P)
        acc += tl.load(
            part_ptr + offs_p[:, None] * ((WIDTH + 1) * D) + offs_n[None, :],
            mask=(offs_p < P)[:, None] & mask_n[None, :],
            other=0.0,
        )
    total = tl.sum(acc, axis=0)
    k = offs_n // D
    d = offs_n % D
    tl.store(dw_ptr + d * WIDTH + k, total.to(dw_ptr.dtype.element_ty), mask=mask_n & (k < WIDTH))
    if HAS_BIAS:
        tl.store(db_ptr + d, total.to(db_ptr.dtype.element_ty), mask=mask_n & (k == WIDTH))


# triton.cdiv and triton.next_power_of_2 are jitted helpers that cost microseconds per host call.
def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


_NUM_SMS: dict[int, int] = {}


def _num_sms(device_index: int) -> int:
    # A plain dict rather than functools.cache, which dynamo warns about.
    if device_index not in _NUM_SMS:
        props = torch.cuda.get_device_properties(device_index)
        _NUM_SMS[device_index] = props.multi_processor_count
    return _NUM_SMS[device_index]


def _check_inputs(x, weight, bias, seq_idx, initial_states):
    assert x.ndim == 3, "x must be (batch, seqlen, dim)"
    B, T, D = x.shape
    assert weight.ndim == 2 and weight.shape[0] == D, "weight must be (dim, width)"
    W = weight.shape[1]
    assert W >= 2, "width must be >= 2"
    assert x.is_cuda and weight.device == x.device
    assert bias is None or bias.shape == (D,)
    assert initial_states is None or initial_states.shape == (B, W - 1, D)
    assert seq_idx is None or seq_idx.shape == (B, T)
    # Offsets within one batch element are computed in int32.
    assert (T + W) * max(x.stride(1), D) < 2**31


def _channels_contiguous(t: Tensor) -> Tensor:
    return t if t.stride(-1) == 1 else t.contiguous()


def _chunking(B, T, D, programs_per_sm, device):
    """(BLOCK_D, n_chunks, chunk) for one-warp programs over 128 channels each: split time into
    chunks until there are about programs_per_sm programs per SM, but keep chunks at least 16 rows
    long so that the PAD halo rows each chunk re-reads stay cheap."""
    block_d = min(128, _next_pow2(D))
    d_blocks = _cdiv(D, block_d)
    sms = _num_sms(device.index)
    n_chunks = max(1, min(T // 16, programs_per_sm * sms // (B * d_blocks)))
    chunk = _cdiv(T, n_chunks)
    return block_d, _cdiv(T, chunk), chunk


# Tuned on an H200. The backward holds more state in registers, which limits it to ~16 warps
# per SM.
_FWD_PROGRAMS_PER_SM = 128
_BWD_PROGRAMS_PER_SM = 16


def _fwd(x, weight, bias, seq_idx, initial_states, final_states, silu):
    B, T, D = x.shape
    W = weight.shape[1]
    out = torch.empty(B, T, D, device=x.device, dtype=x.dtype)
    if out.numel() == 0:
        # Nothing to launch. With T == 0 the final states are the initial states.
        if final_states is not None:
            if initial_states is not None:
                final_states.copy_(initial_states)
            else:
                final_states.zero_()
        return out

    block_d, n_chunks, chunk = _chunking(B, T, D, _FWD_PROGRAMS_PER_SM, x.device)
    # Triton launches on the current device, not the device of the arguments.
    with torch.cuda.device(x.device):
        _fwd_kernel[(n_chunks, _cdiv(D, block_d), B)](
            x,
            weight,
            bias,
            initial_states,
            seq_idx,
            out,
            final_states,
            T,
            D,
            chunk,
            x.stride(0),
            x.stride(1),
            WIDTH=W,
            HAS_BIAS=bias is not None,
            HAS_INIT=initial_states is not None,
            HAS_SEQ=seq_idx is not None,
            HAS_FINAL=final_states is not None,
            SILU=silu,
            BLOCK_D=block_d,
            ROWS=min(8, _next_pow2(chunk)),
            num_warps=1,
        )
    return out


def _bwd(x, weight, bias, seq_idx, initial_states, dout, dfinal_states, need_dinit, silu):
    B, T, D = x.shape
    W = weight.shape[1]
    dx = torch.empty(B, T, D, device=x.device, dtype=x.dtype)
    dinit = torch.empty_like(initial_states) if need_dinit else None
    if dx.numel() == 0:
        # Nothing to launch. With T == 0 the final states are the initial states, and there are no
        # outputs to give weight or bias a gradient.
        if dinit is not None:
            if dfinal_states is not None:
                dinit.copy_(dfinal_states)
            else:
                dinit.zero_()
        dbias = torch.zeros_like(bias) if bias is not None else None
        return dx, torch.zeros_like(weight), dbias, dinit

    block_d, n_chunks, chunk = _chunking(B, T, D, _BWD_PROGRAMS_PER_SM, x.device)
    d_blocks = _cdiv(D, block_d)
    partials = torch.empty(B * n_chunks, W + 1, D, device=x.device, dtype=torch.float32)
    dweight = torch.empty_like(weight)
    dbias = torch.empty_like(bias) if bias is not None else None
    n_cols = (W + (bias is not None)) * D
    # Triton launches on the current device, not the device of the arguments.
    with torch.cuda.device(x.device):
        _bwd_kernel[(n_chunks, d_blocks, B)](
            x,
            weight,
            bias,
            initial_states,
            seq_idx,
            dout,
            dfinal_states,
            dx,
            partials,
            dinit,
            T,
            D,
            chunk,
            x.stride(0),
            x.stride(1),
            dout.stride(0),
            dout.stride(1),
            WIDTH=W,
            HAS_BIAS=bias is not None,
            HAS_INIT=initial_states is not None,
            HAS_SEQ=seq_idx is not None,
            HAS_DFINAL=dfinal_states is not None,
            NEED_DINIT=need_dinit,
            SILU=silu,
            BLOCK_D=block_d,
            ROWS=min(8, _next_pow2(chunk + W - 1)),
            num_warps=1,
        )
        _reduce_partials_kernel[(_cdiv(n_cols, 64),)](
            partials,
            dweight,
            dbias,
            B * n_chunks,
            D,
            n_cols,
            WIDTH=W,
            HAS_BIAS=bias is not None,
            BLOCK_P=32,
            BLOCK_N=64,
            num_warps=4,
        )
    return dx, dweight, dbias, dinit


class _CausalConv1dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, seq_idx, initial_states, return_final_states, silu):
        final_states = None
        if return_final_states:
            final_states = torch.empty(
                x.shape[0], weight.shape[1] - 1, x.shape[2], device=x.device, dtype=x.dtype
            )
        out = _fwd(x, weight, bias, seq_idx, initial_states, final_states, silu)
        ctx.save_for_backward(x, weight, bias, seq_idx, initial_states)
        ctx.silu = silu
        ctx.return_final_states = return_final_states
        return (out, final_states) if return_final_states else out

    @staticmethod
    def backward(ctx, dout, *args):
        x, weight, bias, seq_idx, initial_states = ctx.saved_tensors
        dfinal_states = args[0] if ctx.return_final_states else None
        if dfinal_states is not None:
            dfinal_states = dfinal_states.contiguous()
        need_dinit = initial_states is not None and ctx.needs_input_grad[4]
        dx, dweight, dbias, dinit = _bwd(
            x,
            weight,
            bias,
            seq_idx,
            initial_states,
            _channels_contiguous(dout),
            dfinal_states,
            need_dinit,
            ctx.silu,
        )
        return dx, dweight, dbias, None, dinit, None, None


def causal_conv1d_fn(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    seq_idx: Tensor | None = None,
    initial_states: Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: Tensor | None = None,
    activation: str | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Causal depthwise conv1d over channels-last inputs.

    x: (batch, seqlen, dim); must be contiguous along dim
    weight: (dim, width), width >= 2
    bias: (dim,)
    seq_idx: (batch, seqlen) integer sequence ids for packed sequences. Positions with negative ids
        are considered padding, and give zero outputs. Can be combined with initial_states, which
        count as part of seq_idx[:, 0]'s sequence, but cannot be combined with return_final_states
    initial_states: (batch, width - 1, dim), the inputs preceding x[:, 0]
    return_final_states: also return the last width - 1 rows of concat(initial_states, x), i.e. the
        initial_states for continuing the sequence; supports grad
    final_states_out: optional (batch, width - 1, dim) buffer to copy the final states into
    activation: None, "silu", or "swish"

    Returns out of shape (batch, seqlen, dim) and dtype x.dtype, or (out, final_states).
    """
    if activation not in (None, "silu", "swish"):
        raise NotImplementedError("activation must be None, silu, or swish")
    if seq_idx is not None:
        assert not return_final_states, "cannot return final states if seq_idx is not None"
    if final_states_out is not None:
        assert return_final_states, "final_states_out requires return_final_states=True"
    _check_inputs(x, weight, bias, seq_idx, initial_states)

    x = _channels_contiguous(x)
    weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None
    seq_idx = seq_idx.to(torch.int32).contiguous() if seq_idx is not None else None
    initial_states = initial_states.contiguous() if initial_states is not None else None
    silu = activation is not None

    needs_grad = torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in (x, weight, bias, initial_states)
    )
    if needs_grad:
        result = _CausalConv1dFn.apply(
            x, weight, bias, seq_idx, initial_states, return_final_states, silu
        )
    else:
        final_states = None
        if return_final_states:
            final_states = torch.empty(
                x.shape[0], weight.shape[1] - 1, x.shape[2], device=x.device, dtype=x.dtype
            )
        out = _fwd(x, weight, bias, seq_idx, initial_states, final_states, silu)
        result = (out, final_states) if return_final_states else out

    if final_states_out is not None:
        out, final_states = result
        result = out, final_states_out.copy_(final_states)
    return result
