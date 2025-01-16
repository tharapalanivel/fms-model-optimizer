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
"""Test suite for FMS addon for AIU, introducing GPTQ functionalities"""

# Third Party
import torch

# Local
from fms_mo.aiu_addons.gptq.gptq_aiu_op import register_aiu_gptq_op


def test_gptq_registration() -> None:
    """Call the registration function of GPTQ W4A16 operation, adding the op to torch
    namespace.
    Note: registration must be called before other GPTQ tests that use this op.
    """

    register_aiu_gptq_op()
    assert hasattr(torch.ops, "gptq_gemm")
    assert hasattr(torch.ops.gptq_gemm, "i4f16_fxinputs_aiu")


def test_gptq_op(get_gptq_gemm_inputs: tuple[torch.Tensor, ...]) -> None:
    """Validate output shapes of GPTQ W4A16 tensors.
    Note: this AIU-compatible operation only returns a zero tensor of the
    expected shape, it does not perform a real W4A16 matmul operation.
    """

    x, qweight, qzeros, scales, g_idx = get_gptq_gemm_inputs
    out = torch.ops.gptq_gemm.i4f16_fxinputs_aiu(x, qweight, qzeros, scales, g_idx)
    assert out.size() == torch.Size((x.size()[:-1] + (qweight.size(0),)))
