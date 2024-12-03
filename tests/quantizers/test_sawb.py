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
Test SAWB quantizers
"""

# Third Party
from test_quantizer_utils import quantizer_error, set_base_options
import pytest
import torch

other_option_params = []
for clipSTE in [True, False]:
    for align_zero in [True, False]:
        for use16bins in [True, False]:
            other_option_params.append(
                {"clipSTE": clipSTE, "align_zero": align_zero, "use16bins": use16bins}
            )


@pytest.fixture(params=other_option_params)
def other_options(request):
    """
    Fixture for other options available for SAWB

    Args:
        request (dict): Single Other Option param

    Returns:
        dict: Single Other Option param
    """
    return request.param


# Must override this method for each test file
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
    fms_mo_quantizer.clipSTE = other_option["clipSTE"]
    fms_mo_quantizer.align_zero = other_option["align_zero"]
    fms_mo_quantizer.extended_range = other_option["use16bins"]
    fms_mo_quantizer.set_quantizer()

    # Set n_levels, scale, zero_point for torch_quantizer
    num_bits = torch_quantizer.num_bits

    # For SAWB Zero STEs
    if other_option["align_zero"]:
        # SAWBPlus16ZeroSTE - no sawb_params
        if other_option["clipSTE"] and other_option["use16bins"] and num_bits == 4:
            # Set num_bits and [clip_low,clip_high]
            torch_quantizer.n_levels = 2**num_bits - 1
            torch_quantizer.set_sawb_clip_code(tensor)

            # Scale uses clip_high
            torch_quantizer.scale = (
                torch_quantizer.clip_high * (8.0 / 7.0 + 1.0) / torch_quantizer.n_levels
            )
            torch_quantizer.zero_point = torch.tensor(0)
            torch_quantizer.set_shift_sawb(0)

        # SAWBPlusZeroPerChSTE - TODO: perCh test not functional yet
        elif other_option["clipSTE"] and torch_quantizer.qscheme.q_unit == "perCh":
            pass

        else:  # SAWBZeroSTE, SAWBPlusZeroSTE - sawb_params_code
            # Set num_bits and [clip_low,clip_high]
            torch_quantizer.n_levels = 2**num_bits - 2
            torch_quantizer.set_sawb_clip_code(tensor)

            torch_quantizer.scale = 1.0 / torch_quantizer.n_levels
            torch_quantizer.zero_point = torch.tensor(0)
            torch_quantizer.set_shift_sawb(torch_quantizer.n_levels / 2)
            torch_quantizer.is_single_sided = True
            torch_quantizer.is_symmetric = False

    else:  # SAWBSTE, SAWBPlusSTE ; standard scale,zero_point for [0,1] squashing
        # Set num_bits and [clip_low,clip_high]
        torch_quantizer.n_levels = 2**num_bits - 1
        torch_quantizer.set_sawb_clip(tensor)

        torch_quantizer.scale = 1.0 / torch_quantizer.n_levels
        torch_quantizer.zero_point = torch.tensor(0)
        torch_quantizer.set_shift_sawb(2 ** (num_bits - 1))
        torch_quantizer.is_single_sided = True
        torch_quantizer.is_symmetric = False

    # Set quant_min,quant_max based on n_levels, scale, zero_point
    torch_quantizer.set_quant_range()


def use_tensor_squashing(
    other_option: dict,
    num_bits: int,
    is_perCh: bool,
):
    """
    Specify if tensor squashing is needed (not SAWBPlus16ZeroSTE, SAWBPlusZeroPerChSTE)

    Args:
        other_option (dict): Other Option params
        num_bits (int): Number of bits for quantization
        is_perCh (bool): If qscheme.q_unit == "perCh"

    Returns:
        bool: Specify if tensor needs to be squashed
    """
    is_plus_zero = other_option["clipSTE"] and other_option["align_zero"]
    is_SAWBPlus16ZeroSTE = is_plus_zero and other_option["use16bins"] and num_bits == 4
    is_SAWBPlusZeroPerChSTE = is_plus_zero and is_perCh
    return not (is_SAWBPlus16ZeroSTE or is_SAWBPlusZeroPerChSTE)


# SAWB_new settings for TorchQuantizer ; refactored SAWB does not use tensor squashing
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
    fms_mo_quantizer.clipSTE = other_option["clipSTE"]
    fms_mo_quantizer.align_zero = other_option["align_zero"]
    fms_mo_quantizer.use16bins = other_option["use16bins"]

    fms_mo_quantizer.recompute_clips = True  # Allow computing clips in forward
    fms_mo_quantizer.eval()  # set nn.Module to eval mode
    fms_mo_quantizer.set_quantizer()

    # Set n_levels, scale, zero_point for torch_quantizer
    num_bits = torch_quantizer.num_bits

    # For SAWB Zero STEs
    if other_option["align_zero"]:
        # SAWBPlus16ZeroSTE_new - no sawb_params
        if other_option["clipSTE"] and other_option["use16bins"] and num_bits == 4:
            # Set num_bits and [clip_low,clip_high]
            torch_quantizer.n_levels = 2**num_bits - 1
            torch_quantizer.set_sawb_clip_code(tensor, code=403)  # sets clip_high
            torch_quantizer.scale = (
                torch_quantizer.clip_high * (8.0 / 7.0 + 1.0) / torch_quantizer.n_levels
            )
            torch_quantizer.zero_point = torch.tensor(0)

            # override quant_min calculation w/o resetting qlevel_lowering in qscheme
            torch_quantizer.symmetric_nlevel = 0
            torch_quantizer.set_quant_range()

        # SAWBPlusZeroPerChSTE_new - TODO: perCh test not functional yet
        elif other_option["clipSTE"] and torch_quantizer.qscheme.q_unit == "perCh":
            pass

        else:  # SAWBZeroSTE_new, SAWBPlusZeroSTE_new - sawb_params_code
            torch_quantizer.set_sawb_clip_code(tensor)
            torch_quantizer.set_quant_bounds()

    else:  # SAWBSTE_new, SAWBPlusSTE_new
        # Set num_bits and [clip_low,clip_high]
        torch_quantizer.n_levels = 2**num_bits - 1
        torch_quantizer.set_sawb_clip(tensor)

        torch_quantizer.set_quant_bounds()


##############
# SAWB tests #
##############


def test_sawb_symmetric(
    tensor: torch.FloatTensor,
    sawb_quantizer_symmetric: torch.autograd.Function,
    torch_quantizer_symmetric: torch.nn.Module,
    base_options: dict,
    other_options: dict,
):
    """
    Test SAWB w/ symmetric tensors

    Args:
        tensor (torch.FloatTensor): Tensor to quantize.
        sawb_quantizer_symmetric (torch.autograd.Function): Qmax Quantizer
        torch_quantizer_symmetric (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict): Other Options for quantization.
    """

    # Set base quantizer and PACT2 options ; save nativePT
    native_pt = base_options["nativePT"]
    base_options["nativePT"] = False  # Not supported for SAWB
    set_base_options(sawb_quantizer_symmetric, torch_quantizer_symmetric, base_options)
    # SAWB requires tensor to set parameters for TorchQuantizer
    set_other_options(
        tensor, sawb_quantizer_symmetric, torch_quantizer_symmetric, other_options
    )

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = sawb_quantizer_symmetric(tensor).detach()

    # Squash tensor to [0,1] before forward if needed
    use_squashing = use_tensor_squashing(
        other_options,
        torch_quantizer_symmetric.num_bits,
        torch_quantizer_symmetric.qscheme.q_unit == "perCh",
    )
    if use_squashing:
        tensor2 = torch_quantizer_symmetric.squash_tensor_sawb(tensor)
    else:
        tensor2 = tensor

    qtensor_torch = torch_quantizer_symmetric(tensor2).detach()

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


def test_sawbnew_symmetric(
    tensor: torch.FloatTensor,
    sawbnew_quantizer_symmetric: torch.autograd.Function,
    torch_quantizer_symmetric: torch.nn.Module,
    base_options: dict,
    other_options: dict,
):
    """
    Test SAWB_new w/ symmetric tensors

    Args:
        tensor (torch.FloatTensor): Tensor to quantize.
        sawbnew_quantizer_symmetric (torch.autograd.Function): Qmax Quantizer
        torch_quantizer_symmetric (torch.nn.Module): Torch Quantizer
        base_options (dict): Base options for quantization.
        other_options (dict): Other Options for quantization.
    """
    # Set base quantizer and PACT2 options
    set_base_options(
        sawbnew_quantizer_symmetric, torch_quantizer_symmetric, base_options
    )
    # SAWB requires tensor to set parameters for TorchQuantizer

    # Set TorchQuantizer qlevel_lowering based on align_zero as temp
    qlevel_lowering = torch_quantizer_symmetric.qscheme.qlevel_lowering
    torch_quantizer_symmetric.qscheme.qlevel_lowering = other_options[
        "align_zero"
    ]  # use legacy setting for qlevel_lowering
    set_other_options_new(
        tensor, sawbnew_quantizer_symmetric, torch_quantizer_symmetric, other_options
    )

    # Create quantized tensors from FMS Model Optimizer + torch
    qtensor_fms_mo = sawbnew_quantizer_symmetric(tensor).detach()
    qtensor_torch = torch_quantizer_symmetric(tensor).detach()

    setup = torch_quantizer_symmetric.get_setup()

    # There should be no differences between these two tensors
    quantizer_error(
        tensor, qtensor_fms_mo, qtensor_torch, setup, base_options, other_options
    )
    torch_quantizer_symmetric.qscheme.qlevel_lowering = qlevel_lowering  # reset value
