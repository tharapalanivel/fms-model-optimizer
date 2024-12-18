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
"""Pytest configuration file with fixtures for add-ons functionality testing"""

# Third Party
import pytest
import torch

# ================================================
# GPTQ W4A16 fixtures
# ================================================

gptq_input_sizes = [
    {
        "bs": 4,
        "seq_len": 5,
        "hid_dim": 256,
        "out_feat": 512,
        "n_grp": 4,
    },
]


@pytest.fixture(scope="session", params=gptq_input_sizes)
def get_gptq_gemm_inputs(request) -> tuple[torch.Tensor, ...]:
    """pytest fixture returning test inputs for GPTQ op"""

    sizes = request.param
    compression_factor = 8  # assume 4-bits compression

    x = torch.randn(
        (sizes["bs"], sizes["seq_len"], sizes["hid_dim"]), dtype=torch.float16
    )
    qweight = torch.randint(
        low=0,
        high=torch.iinfo(torch.int32).max,
        size=(sizes["out_feat"], sizes["hid_dim"] // compression_factor),
        dtype=torch.int32,
    )
    qzeros = 8 * torch.ones(
        (sizes["n_grp"], sizes["out_feat"] // compression_factor),
        dtype=torch.int32,
    )
    scales = torch.randn(
        (sizes["n_grp"], sizes["out_feat"]),
        dtype=torch.float16,
    )
    g_idx = torch.zeros(sizes["hid_dim"], dtype=torch.int32)

    return (x, qweight, qzeros, scales, g_idx)


# ================================================
# INT8xINT8 fixtures
# ================================================

i8i8_metadata = [
    {
        "bs": 4,
        "seq_len": 7,
        "hid_dim": 256,
        "out_feat": 512,
        "dtype": torch.float16,
        "wtype": "per_tensor",  # per_channel
        "atype": "per_tensor_symm",  # per_tensor_asymm, per_token
        "smoothquant": False,
    }
]


@pytest.fixture(scope="session", params=i8i8_metadata)
def get_i8i8_gemm_inputs(
    request,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str, bool]:
    """pytest fixture returning test inputs for INT8xINT8 op"""

    data = request.param
    x = torch.randn(
        (data["bs"], data["seq_len"], data["hid_dim"]),
        dtype=data["dtype"],
    ).clamp(-1, 1)
    w_int = torch.randint(
        low=-8,
        high=8,
        size=(data["out_feat"], data["hid_dim"]),
        dtype=torch.int8,
    )
    b = torch.zeros(data["out_feat"], dtype=data["dtype"])
    qdata = create_qdata(
        data["wtype"],
        data["atype"],
        data["hid_dim"],
        data["out_feat"],
        data["smoothquant"],
        data["dtype"],
    )

    return (x, w_int, b, qdata, data["wtype"], data["atype"], data["smoothquant"])


def create_qdata(
    wtype: str,
    atype: str,
    in_feat: int,
    out_feat: int,
    smoothquant: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate dummy qdata tensor based on the provided quantization configuration"""

    qdata_len = 2 if wtype == "per_tensor" else 2 * out_feat  # weight clips
    qdata_len += 2  # activation clips
    qdata_len += out_feat if atype == "per_tensor_asymm" else 1  # zero shift
    qdata_len += in_feat if smoothquant else 1  # smoothquant scales

    # TODO: improve dummy generation
    qdata = torch.ones(qdata_len, dtype=dtype)
    qdata[1] = -qdata[0]  # !!! temporary solution to enforce clip symmetry
    qdata[3] = -qdata[2]
    return qdata
