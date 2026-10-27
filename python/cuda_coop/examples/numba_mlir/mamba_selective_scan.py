# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run the Mamba v1 selective scan with one block per (batch, channel) row.
Each time step is an affine map packed into a uint64; a block-wide inclusive
scan composes them, and a carry links successive tiles.
"""

from __future__ import annotations

import numpy as np
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.cuda import libdevice
from numba_cuda_mlir.cuda.experimental import consteval

import cuda.coop.numba_mlir as coop

# consteval needs a compile-time integer and cannot read a kernel argument, so
# the item count is a module constant; edit it to change the configuration.
THREADS = 256
ITEMS_PER_THREAD = 8
MAX_STATE = 256


@cuda.jit(device=True)
def pack_pair(a, b):
    lo = types.uint64(types.uint32(libdevice.float_as_int(a)))
    hi = types.uint64(types.uint32(libdevice.float_as_int(b)))
    return lo | (hi << types.uint64(32))


@cuda.jit(device=True)
def pair_a(pair):
    return libdevice.int_as_float(types.int32(types.uint32(pair)))


@cuda.jit(device=True)
def pair_b(pair):
    return libdevice.int_as_float(
        types.int32(types.uint32(pair >> types.uint64(32)))
    )


@cuda.jit(device=True)
def apply_softplus(s):
    """Softplus with the threshold at 20: log(1 + exp(s)) for s <= 20, else s"""
    if s > types.float32(20.0):
        return s
    return libdevice.log1pf(libdevice.expf(s))


@cuda.jit(device=True)
def seq_index(row, length, timestep):
    return row * length + timestep


@cuda.jit(device=True)
def a_index(channel, nstate, n):
    return channel * nstate + n


@cuda.jit(device=True)
def state_index(row, nstate, n):
    return row * nstate + n


@cuda.jit(device=True)
def weight_index(
    batch, channel, n, timestep, dim, nstate, length, groups, variable
):
    if not variable:
        return channel * nstate + n
    g = channel // (dim // groups)
    return ((batch * groups + g) * nstate + n) * length + timestep


@cuda.jit(device=True)
def silu(x):
    return x / (types.float32(1.0) + libdevice.expf(-x))


@cuda.jit(device=True)
def compose(left, right):
    """Compose two packed affine maps, applying left first.

    @Returns:
        uint64: The coefficients of the combined affine transformation.
    """
    a0 = libdevice.int_as_float(np.int32(np.uint32(left)))
    b0 = libdevice.int_as_float(np.int32(np.uint32(left >> np.uint64(32))))
    a1 = libdevice.int_as_float(np.int32(np.uint32(right)))
    b1 = libdevice.int_as_float(np.int32(np.uint32(right >> np.uint64(32))))
    alpha = a1 * a0
    beta = a1 * b0 + b1
    alpha_int64 = types.uint64(types.uint32(libdevice.float_as_int(alpha)))
    beta_int64 = types.uint64(types.uint32(libdevice.float_as_int(beta)))
    return alpha_int64 | (beta_int64 << types.uint64(32))


@cuda.jit
def selective_scan_kernel(
    u,
    delta,
    a,
    b,
    c,
    d,
    z,
    bias,
    output,
    last_state,
    dim,
    length,
    nstate,
    b_groups,
    c_groups,
    variable_b,
    variable_c,
    has_d,
    has_z,
    has_bias,
    softplus,
):
    block = coop.this_block()
    thread = cuda.threadIdx.x
    row = cuda.blockIdx.x
    batch, channel = row // dim, row % dim
    u_td = coop.ThreadData(items_per_thread=ITEMS_PER_THREAD)
    delta_td = coop.ThreadData(items_per_thread=ITEMS_PER_THREAD)
    b_td = coop.ThreadData(items_per_thread=ITEMS_PER_THREAD)
    c_td = coop.ThreadData(items_per_thread=ITEMS_PER_THREAD)
    pairs = coop.ThreadData(items_per_thread=ITEMS_PER_THREAD)
    agg = coop.ThreadData(items_per_thread=1)
    y_td = coop.ThreadData(items_per_thread=ITEMS_PER_THREAD)

    h_carry = cuda.shared.array(MAX_STATE, types.float32)
    for n in range(thread, nstate, THREADS):
        h_carry[n] = types.float32(0.0)

    block.sync_aligned()

    tile_items = THREADS * ITEMS_PER_THREAD
    ntiles = (length + tile_items - 1) // tile_items

    for tile in range(ntiles):
        start = tile * tile_items
        valid = min(length - start, tile_items)

        coop.load(
            block,
            u,
            u_td,
            offset=seq_index(row, length, start),
            algorithm="warp_transpose",
            valid_items=valid,
            oob_default=0,
        )
        coop.load(
            block,
            delta,
            delta_td,
            offset=seq_index(row, length, start),
            algorithm="warp_transpose",
            valid_items=valid,
            oob_default=0,
        )

        for i in consteval(range(ITEMS_PER_THREAD)):
            s = delta_td[i]
            if has_bias:
                s = s + bias[channel]
            if softplus:
                s = apply_softplus(s)
            delta_td[i] = types.float32(s)
            y_td[i] = types.float32(0.0)

        for n in range(nstate):
            A_value = a[a_index(channel, nstate, n)]
            h_in = h_carry[n]

            B_fixed = types.float32(0.0)
            if variable_b:
                coop.load(
                    block,
                    b,
                    b_td,
                    offset=weight_index(
                        batch,
                        channel,
                        n,
                        start,
                        dim,
                        nstate,
                        length,
                        b_groups,
                        variable_b,
                    ),
                    algorithm="warp_transpose",
                    valid_items=valid,
                    oob_default=0,
                )
            else:
                B_fixed = b[a_index(channel, nstate, n)]
            C_fixed = types.float32(0.0)
            if variable_c:
                coop.load(
                    block,
                    c,
                    c_td,
                    offset=weight_index(
                        batch,
                        channel,
                        n,
                        start,
                        dim,
                        nstate,
                        length,
                        c_groups,
                        variable_c,
                    ),
                    algorithm="warp_transpose",
                    valid_items=valid,
                    oob_default=0,
                )
            else:
                C_fixed = c[a_index(channel, nstate, n)]

            for i in consteval(range(ITEMS_PER_THREAD)):
                t = start + thread * ITEMS_PER_THREAD + i
                if t < length:
                    B_value = B_fixed
                    if variable_b:
                        B_value = b_td[i]
                    alpha = libdevice.expf(delta_td[i] * A_value)
                    beta = delta_td[i] * B_value * u_td[i]
                    pairs[i] = types.uint64(pack_pair(alpha, beta))
                else:
                    pairs[i] = types.uint64(
                        pack_pair(types.float32(1.0), types.float32(0.0))
                    )

            scan_result = coop.inclusive_scan(
                block,
                pairs,
                scan_op=compose,
                algorithm="warp_scans",
                aggregate_output=agg,
            )
            for i in consteval(range(ITEMS_PER_THREAD)):
                t = start + thread * ITEMS_PER_THREAD + i
                if t < length:
                    C_value = C_fixed
                    if variable_c:
                        C_value = c_td[i]
                else:
                    C_value = types.float32(0.0)
                p = scan_result[i]
                h = pair_a(p) * h_in + pair_b(p)
                y_td[i] = types.float32(y_td[i] + C_value * h)

            h_new = pair_a(agg[0]) * h_in + pair_b(agg[0])
            if thread == 0:
                h_carry[n] = h_new
            block.sync_aligned()

        for i in consteval(range(ITEMS_PER_THREAD)):
            t = start + thread * ITEMS_PER_THREAD + i
            v = y_td[i]
            if has_d:
                v += d[channel] * u_td[i]
            if t < length and has_z:
                v *= silu(z[seq_index(row, length, t)])
            y_td[i] = types.float32(v)

        coop.store(
            block,
            output,
            y_td,
            offset=seq_index(row, length, start),
            algorithm="warp_transpose",
            valid_items=valid,
        )

        block.sync_aligned()

    for n in range(thread, nstate, THREADS):
        last_state[state_index(row, nstate, n)] = h_carry[n]


def selective_scan(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
):
    """Return output and canonical final state for finite float32 inputs.

    u/delta: (batch, dim, length); A: (dim, state), 1 <= state <= 256.
    B/C: (dim, state), (batch, state, length) or (batch, groups, state, length).
    Groups must divide dim. Optional D/bias: (dim,); z: u.shape.
    Each block of THREADS threads scans tiles of THREADS * ITEMS_PER_THREAD
    time steps.
    This teaching wrapper copies arrays and synchronizes each invocation.
    """

    def checked(name, value, shape=None):
        value = np.asarray(value)
        if value.dtype != np.float32:
            raise TypeError(f"{name} must have dtype float32")
        if shape is not None and value.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {value.shape}"
            )
        return np.ascontiguousarray(value)

    u = checked("u", u)
    if u.ndim != 3 or any(s < 1 for s in u.shape):
        raise ValueError("u must have positive (batch, dim, length) dimensions")
    batch, dim, length = u.shape
    delta = checked("delta", delta, u.shape)
    A = checked("A", A)
    if A.ndim != 2 or A.shape[0] != dim or not 1 <= A.shape[1] <= MAX_STATE:
        raise ValueError(
            "A must have shape (dim, state), with 1 <= state <= 256"
        )
    nstate = A.shape[1]

    def weights(name, value):
        value = checked(name, value)
        if value.shape == (dim, nstate):
            return value, False, 1
        if value.shape == (batch, nstate, length):
            return value, True, 1
        if (
            value.ndim == 4
            and value.shape[0] == batch
            and value.shape[2:] == (nstate, length)
            and value.shape[1] > 0
            and dim % value.shape[1] == 0
        ):
            return value, True, value.shape[1]
        raise ValueError(f"unsupported {name} shape {value.shape}")

    B, variable_b, b_groups = weights("B", B)
    C, variable_c, c_groups = weights("C", C)
    optional = []
    for name, value, shape in (
        ("D", D, (dim,)),
        ("z", z, u.shape),
        ("delta_bias", delta_bias, (dim,)),
    ):
        optional.append(
            np.zeros(1, dtype=np.float32)
            if value is None
            else checked(name, value, shape)
        )
    host_inputs = [u, delta, A, B, C, *optional]
    device_inputs = [cuda.to_device(value.reshape(-1)) for value in host_inputs]
    output = cuda.device_array(u.size, dtype=np.float32)
    last = cuda.device_array(batch * dim * nstate, dtype=np.float32)
    selective_scan_kernel[batch * dim, THREADS](
        *device_inputs,
        output,
        last,
        dim,
        length,
        nstate,
        b_groups,
        c_groups,
        variable_b,
        variable_c,
        D is not None,
        z is not None,
        delta_bias is not None,
        bool(delta_softplus),
    )
    return output.copy_to_host().reshape(u.shape), last.copy_to_host().reshape(
        batch, dim, nstate
    )


def _expand_weight(weight, batch, dim, nstate, length):
    """Broadcast B or C to (batch, dim, nstate, length)."""
    if weight.shape == (dim, nstate):
        return np.broadcast_to(
            weight[None, :, :, None], (batch, dim, nstate, length)
        )
    if weight.ndim == 3:
        return np.broadcast_to(weight[:, None], (batch, dim, nstate, length))
    channels_per_group = dim // weight.shape[1]
    return weight[:, np.arange(dim) // channels_per_group]


def selective_scan_reference(
    u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False
):
    """Serial float64 reference; same arguments as ``selective_scan``."""
    u = u.astype(np.float64)
    batch, dim, length = u.shape
    A = A.astype(np.float64)
    nstate = A.shape[1]
    s = delta.astype(np.float64)
    if delta_bias is not None:
        s = s + delta_bias.astype(np.float64)[None, :, None]
    if delta_softplus:
        s = np.where(s <= 20.0, np.log1p(np.exp(np.minimum(s, 20.0))), s)
    B = _expand_weight(B.astype(np.float64), batch, dim, nstate, length)
    C = _expand_weight(C.astype(np.float64), batch, dim, nstate, length)

    alpha = np.exp(s[:, :, None, :] * A[None, :, :, None])
    beta = (s * u)[:, :, None, :] * B
    h = np.zeros((batch, dim, nstate))
    v = np.empty_like(u)
    for t in range(length):
        h = alpha[..., t] * h + beta[..., t]
        v[..., t] = np.sum(C[..., t] * h, axis=-1)
    if D is not None:
        v = v + D.astype(np.float64)[None, :, None] * u
    if z is not None:
        z = z.astype(np.float64)
        v = v * z / (1.0 + np.exp(-z))
    return v, h


def run_example() -> float:
    """Check lengths, states, B/C layouts, and options; return the max error."""

    rng = np.random.default_rng(0)
    batch, dim, groups = 2, 4, 2
    tile_items = THREADS * ITEMS_PER_THREAD

    def weight(layout, nstate, length):
        shape = {
            "fixed": (dim, nstate),
            "variable": (batch, nstate, length),
            "grouped": (batch, groups, nstate, length),
        }[layout]
        return rng.standard_normal(shape, dtype=np.float32)

    configs = [
        (length, 1, "fixed", "fixed", False)
        for length in (31, tile_items, tile_items + 1, 2 * tile_items + 1)
    ] + [
        (tile_items + 1, 3, "variable", "variable", False),
        (tile_items + 1, 16, "grouped", "grouped", False),
        (2 * tile_items + 1, 8, "fixed", "grouped", False),
        (31, 4, "variable", "fixed", True),
        (2 * tile_items + 1, 16, "grouped", "variable", True),
    ]
    worst = 0.0
    for length, nstate, b_layout, c_layout, options in configs:
        shape = (batch, dim, length)
        u = rng.standard_normal(shape, dtype=np.float32)
        A = -rng.uniform(0.5, 2.0, (dim, nstate)).astype(np.float32)
        B = weight(b_layout, nstate, length)
        C = weight(c_layout, nstate, length)
        if options:
            # Pre-activation delta: softplus maps it to small positive steps.
            delta = rng.normal(-3.0, 1.0, shape).astype(np.float32)
            extra = {
                "D": rng.standard_normal(dim, dtype=np.float32),
                "z": rng.standard_normal(shape, dtype=np.float32),
                "delta_bias": rng.uniform(-0.5, 0.5, dim).astype(np.float32),
                "delta_softplus": True,
            }
        else:
            delta = rng.uniform(0.01, 0.1, shape).astype(np.float32)
            extra = {}

        output, last_state = selective_scan(u, delta, A, B, C, **extra)
        want_output, want_state = selective_scan_reference(
            u, delta, A, B, C, **extra
        )
        np.testing.assert_allclose(output, want_output, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(last_state, want_state, rtol=1e-4, atol=1e-5)
        worst = max(worst, float(np.max(np.abs(output - want_output))))
    return worst


def main() -> int:
    error = run_example()
    print(
        f"threads={THREADS}, items_per_thread={ITEMS_PER_THREAD}: "
        f"max output error {error:.2e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
