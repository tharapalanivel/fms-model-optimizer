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

"""
Test Qmax quantizers
"""

# Third Party
from test_quantizer_utils import quantizer_error, set_base_options
import pytest
import torch

other_options_params = []
# False causes errors due to decimal zero_point variances w/ linear_quantize
# vs torch.fake_quantize_per_tensor_affine
for align_zero in [True]:
    for minmax in [True, False]:
        for extend_act_range in [True, False]:
            # Only minmax or extend_act_range can be used at once
            if not (minmax and extend_act_range):
                other_options_params.append(
                    {
                        "align_zero": align_zero,
                        "minmax": minmax,
                        "extend_act_range": extend_act_range,
                    }
                )


@pytest.fixture(params=other_options_params)
def other_options(request):
    """
    Fixture for other options available for Qmax

    Args:
        request (dict): Single Other Option param

    Returns:
        dict: Single Other Option param
    """
    return request.param


def set_other_options(
    tensor: torch.FloatTensor,
    fms_mo_quantizer: torch.autograd.Function,
    torch_quantizer: torch.nn.Module,
    other_option: dict,
):
    """
    Set other options for FMS and Torch quantizer

    Args:
        tensor (torch.FloatTensor): Tensor to quantize.
        fms_mo_quantizer (torch.autograd.Function):  FMS quantizer
        torch_quantizer (torch.nn.Module): Torch Quantizer
        other_option (dict): Other Option params
    """
    fms_mo_quantizer.align_zero = other_option["align_zero"]
    fms_mo_quantizer.minmax = other_option["minmax"]
    fms_mo_quantizer.extend_act_range = other_option["extend_act_range"]

    fms_mo_quantizer.set_quantizer()

    q_unit = torch_quantizer.qscheme.q_unit
    num_bits = torch_quantizer.num_bits
    is_perCh = q_unit == "perCh"
    is_perGp = q_unit == "perGp"

    n_half = 2 ** (num_bits - 1)

    if other_option["minmax"]:
        if is_perCh:  # QminmaxPerChSTE
            pass
        elif is_perGp:  # QminmaxPerGpSTE
            pass
        else:  # QminmaxSTE
            # min/max clip vals asymmetric
            torch_quantizer.clip_low = tensor.min()
            torch_quantizer.clip_high = tensor.max()

            torch_quantizer.n_levels = 2**num_bits - 1
            torch_quantizer.scale = (
                torch_quantizer.clip_high - torch_quantizer.clip_low
            ) / torch_quantizer.n_levels
            torch_quantizer.zero_point = torch.round(
                -torch_quantizer.clip_low / torch_quantizer.scale
            ).to(torch.int)
            torch_quantizer.set_shift_sawb(n_half)
            torch_quantizer.is_single_sided = False

    elif other_option["extend_act_range"]:  # QmaxExtendRangeSTE
        clip_ratio = -n_half / (n_half - 1)  # -128/127, -8/7
        if tensor.max() >= tensor.min().abs():
            torch_quantizer.clip_high = tensor.max()
            torch_quantizer.clip_low = torch_quantizer.clip_high * clip_ratio
        else:
            torch_quantizer.clip_low = tensor.min()
            torch_quantizer.clip_high = torch_quantizer.clip_low / clip_ratio

        torch_quantizer.n_levels = 2**num_bits - 2
        torch_quantizer.scale = torch_quantizer.clip_high * 2 / torch_quantizer.n_levels
        torch_quantizer.zero_point = torch.tensor([0])
        torch_quantizer.set_shift_sawb(0)
        torch_quantizer.is_single_sided = False
        torch_quantizer.symmetric_nlevel = 0

    else:  # non-minmax STEs
        if is_perCh:  # QmaxPerChSTE
            pass
        elif is_perGp:  # QmaxPerGpSTE
            pass
        else:  # QmaxSTE
            # clip_vals symmetric
            torch_quantizer.clip_high = tensor.abs().max()
            torch_quantizer.clip_low = -torch_quantizer.clip_high

            torch_quantizer.n_levels = 2**num_bits - 2
            torch_quantizer.scale = torch.tensor([1.0 / torch_quantizer.n_levels])
            torch_quantizer.zero_point = torch.tensor(0)
            torch_quantizer.set_shift_sawb(torch_quantizer.n_levels / 2)  # 2**(b-1)-1
            torch_quantizer.is_single_sided = True  # [-clip,clip] -> [0,1]
            torch_quantizer.is_symmetric = False

    # Set quant_min,quant_max based on n_levels, scale, zero_point
    torch_quantizer.set_quant_range()


def use_tensor_squashing(other_option, q_unit):
    """
    Specify if tensor squashing is needed (ie using QmaxSTE)

    Args:
        other_option (dict): Other Option params
        q_unit (str): Qscheme q_unit variable

    Returns:
        bool: Specify if tensor needs to be squashed
    """
    is_perCh = q_unit == "perCh"
    is_perGp = q_unit == "perGp"
    # Only QmaxSTE uses tensor squashing
    return not (
        other_option["minmax"]
        or other_option["extend_act_range"]
        or is_perCh
        or is_perGp
    )


def set_other_options_new(
    tensor: torch.FloatTensor,
    fms_mo_quantizer: torch.autograd.Function,
    torch_quantizer: torch.nn.Module,
    other_option: dict,
):
    """
    Set other options for new FMS and Torch quantizer

    Args:
        tensor (torch.FloatTensor): Tensor to quantize.
        fms_mo_quantizer (torch.autograd.Function):  FMS quantizer
        torch_quantizer (torch.nn.Module): Torch Quantizer
        other_option (dict): Other Option params
    """
    fms_mo_quantizer.align_zero = other_option["align_zero"]
    fms_mo_quantizer.minmax = other_option["minmax"]
    fms_mo_quantizer.extend_act_range = other_option["extend_act_range"]
    fms_mo_quantizer.recompute_clips = (
        True  # Allow forward method to compute clips per call
    )
    fms_mo_quantizer.eval()  # set nn.Module to eval mode

    fms_mo_quantizer.set_quantizer()

    q_unit = torch_quantizer.qscheme.q_unit
    num_bits = torch_quantizer.num_bits
    is_perCh = q_unit == "perCh"
    is_perGp = q_unit == "perGp"

    n_half = 2 ** (num_bits - 1)

    if other_option["minmax"]:
        if is_perCh:  # QminmaxPerChSTE_new
            pass
        elif is_perGp:  # QminmaxPerGpSTE_new
            pass
        else:  # QminmaxSTE_new
            # min/max clip vals asymmetric
            torch_quantizer.clip_low = tensor.min()
            torch_quantizer.clip_high = tensor.max()

            torch_quantizer.n_levels = 2**num_bits - 1
            torch_quantizer.scale = (
                torch_quantizer.clip_high - torch_quantizer.clip_low
            ) / torch_quantizer.n_levels
            torch_quantizer.zero_point = torch.round(
                -torch_quantizer.clip_low / torch_quantizer.scale
            ).to(torch.int)
            torch_quantizer.is_symmetric = False  # asymmetric

    elif other_option["extend_act_range"]:  # QmaxExtendRangeSTE_new
        clip_ratio = -n_half / (n_half - 1)  # -128/127, -8/7
        if tensor.max() >= tensor.min().abs():
            torch_quantizer.clip_high = tensor.max()
            torch_quantizer.clip_low = torch_quantizer.clip_high * clip_ratio
        else:
            torch_quantizer.clip_low = tensor.min()
            torch_quantizer.clip_high = torch_quantizer.clip_low / clip_ratio

        torch_quantizer.n_levels = 2**num_bits - 2
        torch_quantizer.scale = torch_quantizer.clip_high * 2 / torch_quantizer.n_levels
        torch_quantizer.zero_point = torch.tensor(0)
        torch_quantizer.symmetric_nlevel = 0

    else:  # non-minmax STEs
        if is_perCh:  # QmaxPerChSTE_new
            pass
        elif is_perGp:  # QmaxPerGpSTE_new
            pass
        else:  # QmaxSTE_new
            # clip_vals symmetric
            torch_quantizer.n_levels = (
                2**num_bits - 2
                if torch_quantizer.qscheme.qlevel_lowering
                else 2**num_bits - 1
            )
            torch_quantizer.clip_high = tensor.abs().max()
            torch_quantizer.clip_low = -torch_quantizer.clip_high

            torch_quantizer.scale = (
                torch_quantizer.clip_high * 2 / torch_quantizer.n_levels
            )
            torch_quantizer.zero_point = torch.tensor(0)

    # Set quant_min,quant_max based on n_levels, scale, zero_point
    torch_quantizer.set_quant_range()


##############
# Qmax tests #
##############


def test_qmax_symmetric(
    tensor: torch.FloatTensor,
    qmax_quantizer_symmetric: torch.autograd.Function,
    torch_quantizer_symmetric: torch.nn.Module,
    base_options: dict,
    other_options: dict,
):
    """
    Test Qmax w/ symmetric tensors

    Args:
        tensor (torch.FloatTensor): Tensor to quantize.
        qmax_quantizer_symmetric (torch.autograd.Function): Qmax Quantizer
        torch_quantizer_symmetric (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict): Other Options for quantization.
    """
    # Set base quantizer and PACT2 options
    native_pt = base_options["nativePT"]  # save nativePT
    base_options["nativePT"] = False  # Not supported for Qmax
    set_base_options(qmax_quantizer_symmetric, torch_quantizer_symmetric, base_options)
    # SAWB requires tensor to set parameters for TorchQuantizer
    set_other_options(
        tensor, qmax_quantizer_symmetric, torch_quantizer_symmetric, other_options
    )

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = qmax_quantizer_symmetric(tensor).detach()

    # Squash tensor to [0,1] before forward
    use_squashing = use_tensor_squashing(
        other_options, torch_quantizer_symmetric.qscheme.q_unit
    )
    if use_squashing:
        tensor = torch_quantizer_symmetric.squash_tensor_sawb(tensor)

    qtensor_torch = torch_quantizer_symmetric(tensor).detach()

    if base_options["dequantize"]:
        # Unsquash tensor to [-clip,clip] after forward for floats and symmetric qlevels for ints
        if use_squashing:
            qtensor_torch = torch_quantizer_symmetric.unsquash_tensor_sawb(
                qtensor_torch
            )
    else:
        qtensor_torch = torch_quantizer_symmetric.shift_qtensor_sawb(qtensor_torch)

    setup = torch_quantizer_symmetric.get_setup()

    # There should be no differences between these two tensors
    quantizer_error(
        tensor, qtensor_fms_mo, qtensor_torch, setup, base_options, other_options
    )
    base_options["nativePT"] = native_pt  # reset value


def test_qmaxnew_symmetric(
    tensor: torch.FloatTensor,
    qmaxnew_quantizer_symmetric: torch.autograd.Function,
    torch_quantizer_symmetric: torch.nn.Module,
    base_options: dict,
    other_options: dict,
):
    """
    Test Qmax_new w/ symmetric tensors

    Args:
        tensor (torch.FloatTensor): Tensor to quantize.
        qmaxnew_quantizer_symmetric (torch.autograd.Function): Qmax Quantizer
        torch_quantizer_symmetric (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict): Other Options for quantization.
    """
    # Set base quantizer and PACT2 options
    set_base_options(
        qmaxnew_quantizer_symmetric, torch_quantizer_symmetric, base_options
    )
    # SAWB requires tensor to set parameters for TorchQuantizer
    set_other_options_new(
        tensor, qmaxnew_quantizer_symmetric, torch_quantizer_symmetric, other_options
    )

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = qmaxnew_quantizer_symmetric(tensor).detach()
    qtensor_torch = torch_quantizer_symmetric(tensor).detach()

    setup = torch_quantizer_symmetric.get_setup()

    # There should be no differences between these two tensors
    quantizer_error(
        tensor, qtensor_fms_mo, qtensor_torch, setup, base_options, other_options
    )
