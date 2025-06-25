# Copyright The FMS Model Optimizer Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FMS registration of attention BMM operation using torch-registered scaled BMM."""

# Standard
from importlib.util import find_spec
from typing import NotRequired, Unpack
import math

# Third Party
from fms.modules.attention import (
    AttentionKwargs,
    _sdpa_update_attn_kwargs,
    register_attention_op,
)
from torch import Tensor
import torch

# Local
import fms_mo.aiu_addons.fp8.fp8_aiu_op  # pylint: disable=unused-import

if find_spec("torchao"):
    TORCHAO_INSTALLED = True
    # Third Party
    from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor
    from torchao.dtypes.floatx.float8_layout import (
        Float8AQTTensorImpl,
        Float8Layout,
        Float8MMConfig,
    )
    from torchao.quantization.granularity import PerTensor
    from torchao.quantization.observer import get_block_size
    from torchao.quantization.quant_primitives import ZeroPointDomain
else:
    TORCHAO_INSTALLED = False


class MathFP8AttentionKwargs(AttentionKwargs):
    """TypedDict for FP8 attention."""

    mask: NotRequired[Tensor]
    do_scale_q: bool
    is_causal_mask: bool


# TODO: Doesn't quite work yet, more discussion needed
Q_RANGE = 200.0
K_RANGE = 200.0
V_RANGE = 100.0


def _construct_fp8_cache(
    tensor: Tensor, scale: Tensor, orig_dtype: torch.dtype
) -> AffineQuantizedTensor:
    """Construct the torchao tensor to save kv cache with its scales."""

    weight_granularity = PerTensor()
    fp8_layout = Float8Layout(Float8MMConfig(use_fast_accum=True))
    return AffineQuantizedTensor(
        Float8AQTTensorImpl.from_plain(
            tensor,
            scale,
            None,
            fp8_layout,
        ),
        get_block_size(tensor.shape, weight_granularity),
        tensor.shape,
        zero_point_domain=ZeroPointDomain.NONE,
        dtype=orig_dtype,
    )


def _math_fp8_store_op(
    keys: Tensor,  # pylint: disable=unused-argument
    values: Tensor,
    key_cache: Tensor | None,
    value_cache: Tensor | None,
    **attn_kwargs: Unpack[MathFP8AttentionKwargs],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Implement math of KV cache storing."""

    orig_dtype = keys.dtype

    if isinstance(key_cache, AffineQuantizedTensor) and isinstance(
        value_cache, AffineQuantizedTensor
    ):
        k_scale = key_cache.tensor_impl.scale
        v_scale = value_cache.tensor_impl.scale
    else:
        k_scale = (torch.abs(keys).max() / K_RANGE).to(dtype=torch.float32)
        v_scale = (torch.abs(values).max() / V_RANGE).to(dtype=torch.float32)

    keys = (keys / k_scale).to(torch.float8_e4m3fn).transpose(2, 1)
    values = (values / v_scale).to(torch.float8_e4m3fn).transpose(2, 1)

    if (
        isinstance(key_cache, AffineQuantizedTensor)
        and isinstance(value_cache, AffineQuantizedTensor)
        and value_cache.numel() > 0
    ):
        key_cache = torch.cat((key_cache.tensor_impl.float8_data, keys), dim=2)
        value_cache = torch.cat((value_cache.tensor_impl.float8_data, values), dim=2)
        key_cache = _construct_fp8_cache(key_cache, k_scale, orig_dtype)
        value_cache = _construct_fp8_cache(value_cache, v_scale, orig_dtype)
        return (
            key_cache,
            value_cache,
            key_cache,
            value_cache,
        )

    keys = _construct_fp8_cache(keys, k_scale, orig_dtype)
    values = _construct_fp8_cache(values, v_scale, orig_dtype)
    return (keys, values, keys, values)


def _math_fp8_compute_op(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    nheads: int,
    kvheads: int,
    p_dropout: float,
    scale_factor: float | None,
    **attn_kwargs: Unpack[MathFP8AttentionKwargs],
) -> Tensor:
    """Implement computation of attention BMM, leveraging the custom scaled attention
    BMM op that was pre-registered for torch.compile."""

    orig_dtype = query.dtype

    q_scale = torch.tensor(1.0, dtype=torch.float32, device=query.device)
    if attn_kwargs.get("do_scale_q", False):
        q_scale.copy_(torch.abs(query).max() / Q_RANGE)
        query = query / q_scale

    query = query.to(torch.float8_e4m3fn).transpose(2, 1)

    if isinstance(key_cache, AffineQuantizedTensor) and isinstance(
        value_cache, AffineQuantizedTensor
    ):
        k_scale = key_cache.tensor_impl.scale
        v_scale = value_cache.tensor_impl.scale
        key_cache = key_cache.tensor_impl.float8_data
        value_cache = value_cache.tensor_impl.float8_data
    else:
        k_scale = (torch.abs(key_cache).max() / K_RANGE).to(dtype=torch.float32)
        v_scale = (torch.abs(value_cache).max() / V_RANGE).to(dtype=torch.float32)
        key_cache = (key_cache / k_scale).to(torch.float8_e4m3fn)
        value_cache = (value_cache / v_scale).to(torch.float8_e4m3fn)

    # no longer transposing prior to store, so need to check this in case of no cache
    # TODO: Refactor FMS to avoid edge cases where this fails; add use_cache param here
    if key_cache.shape[1] != kvheads and key_cache.shape[2] == kvheads:
        key_cache = key_cache.transpose(2, 1)
        value_cache = value_cache.transpose(2, 1)

    mask = attn_kwargs.get("mask", None)
    if mask is not None:
        # Our expected mask format is bs x q_len x k_len, so to make it broadcastable
        # we need to create the nheads dimension
        while len(mask.size()) != 4:  # expects bs (x nheads) x q_len x kv_len
            mask = mask.unsqueeze(1)

    L, S = query.size(-2), key_cache.size(-2)
    scale_factor = (
        1 / math.sqrt(query.size(-1)) if scale_factor is None else scale_factor
    )
    attn_bias = torch.zeros(L, S, dtype=orig_dtype, device=query.device)
    if attn_kwargs.get("is_causal_mask", False):
        assert mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(torch.float32)

    if mask is not None:
        if mask.dtype == torch.bool:
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
        else:
            attn_bias = mask + attn_bias

    expansion = nheads // kvheads
    if expansion > 1:
        key_cache = key_cache.repeat_interleave(
            query.size(-3) // key_cache.size(-3), -3
        )
        value_cache = value_cache.repeat_interleave(
            query.size(-3) // value_cache.size(-3), -3
        )

    attn_weight = (
        torch.ops.sendnn.scaled_bmm(
            query,
            key_cache.transpose(-2, -1),
            q_scale,
            k_scale,
            out_dtype=orig_dtype,
            use_fast_accum=True,
        )
        * scale_factor
    )
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, p_dropout, train=True)
    # Do matmul in orig_dtype
    attn = attn_weight @ (value_cache.to(dtype=orig_dtype) * v_scale)

    attn = attn.to(orig_dtype).transpose(2, 1).contiguous()
    return attn


register_attention_op(
    "math_fp8",
    _math_fp8_store_op,
    _math_fp8_compute_op,
    update_attn_kwargs_op=_sdpa_update_attn_kwargs,
)
