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
"""Test suite for FMS addon for AIU, introducing INT8xINT8 functionalities"""

# Third Party
import torch

# Local
from fms_mo.aiu_addons.i8i8.i8i8_aiu_op import register_aiu_i8i8_op


def test_i8i8_registration() -> None:
    """Call the registration function of INT8xINT8 operation, adding the op to torch
    namespace.
    Note: registration must be called before other INT8 tests that use this op.
    """

    register_aiu_i8i8_op()
    assert hasattr(torch.ops, "fms_mo")
    assert hasattr(torch.ops.fms_mo, "i8i8_aiu")


def test_i8i8_op(
    get_i8i8_gemm_inputs: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str, bool
    ],
) -> None:
    """Validate output shapes of INT8xINT8 matmul.
    Computations are simulated, using quantized/dequantized tensors.
    """

    (
        x,
        weight,
        bias,
        qdata,
        weight_quant_type,
        activ_quant_type,
        smoothquant,
    ) = get_i8i8_gemm_inputs

    out = torch.ops.fms_mo.i8i8_aiu(
        x,
        weight,
        bias,
        qdata,
        weight_quant_type,
        activ_quant_type,
        smoothquant,
    )

    assert out.size() == torch.Size((x.size()[:-1] + (weight.size(0),)))
