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
Base Per Channel Quantizer Class
"""
from typing import List

# Third Party
import torch

# Local
from fms_mo.quant_refactor.linear_utils import (
    asymmetric_linear_quantization_params,
    linear_quantization,
    symmetric_linear_quantization_params,
    transform_clips,
)


class PerChSTEBase(torch.autograd.Function):
    """Base class for customized forward/backward functions that is NOT using PT native func.
    There's a family of non-learnable quantizers, such as SAWB, MinMax,
    whose forward can be the same quant functions and backward is simply STE.
    We just need to calculate scale in the upper level quantizer class then those quantizers
    could all be using the same base "STE function"

    math should be consistent with pytorch: https://pytorch.org/docs/stable/quantization.html
        x_int = round(x/scale + zp)
        x_dq  = (x_int - zp) * scale # TODO check clamp before or after dQ
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.FloatTensor,
        num_bits: torch.IntTensor,
        clip_valn: torch.FloatTensor,
        clip_val: torch.FloatTensor,
        dequantize: bool = True,
        symmetric: bool = False,
        qlevel_lowering: bool = False,
    ):
        """
        General forward method:
            Set clip values to dtype of input_tensor tensor
            Compute # of quantized levels, scale, and zero point
            Save data for backward()
            Perform linear quantization on input_tensor tensor
            return output

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            input_tensor_tensor (torch.FloatTensor): Tensor to be quantized.
            num_bits (torch.IntTensor): Number of bit for quantization.
            clip_valn (torch.FloatTensor): Lower clip value bound.
            clip_val (torch.FloatTensor): Upper clip value bound.
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """
        clip_valn, clip_val = transform_clips(input_tensor.dtype, clip_valn, clip_val)
        n_levels, scale, zero_point = PerChSTEBase.calc_qparams(
            input_tensor, num_bits, clip_valn, clip_val, qlevel_lowering
        )
        PerChSTEBase.save_tensors(
            ctx,
            tensors=(input_tensor, n_levels, clip_valn, clip_val, scale, zero_point),
        )
        output = linear_quantization(
            input_tensor,
            num_bits,
            scale,
            zero_point,
            dequantize,
            symmetric,
            qlevel_lowering,
        )
        return output

    @classmethod
    def calc_qparams(
        cls,
        num_bits: torch.IntTensor,
        clip_valn: torch.FloatTensor,
        clip_val: torch.FloatTensor,
        symmetric: bool = False,
        qlevel_lowering: bool = True,
    ):
        """
        Compute the scale and zero_point from num_bits and clip values

        Args:
            num_bits (torch.IntTensor): Number of bit for quantization.
            clip_valn (torch.FloatTensor): Lower clip value.
            clip_val (torch.FloatTensor): Upper clip value.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.

        Returns:
            [torch.IntTensor, torch.FloatTensor, torch.IntTensor]: Quantized parameters
        """
        if symmetric:
            n_levels, scale, zero_point = symmetric_linear_quantization_params(
                num_bits,
                clip_val,
                qlevel_lowering,
            )
        else:
            n_levels, scale, zero_point = asymmetric_linear_quantization_params(
                num_bits,
                clip_valn,
                clip_val,
                integral_zero_point=True,
                signed=False,
            )
        return n_levels, scale, zero_point


    @staticmethod
    def backward(ctx, grad_output):
        """
        General STE backward method:
            Return grad_output + None args to match forward input_tensors

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            grad_output (torch.FloatTensor): Gradient tensor

        Returns:
            [torch.FloatTensor, None,...,None]: STE Gradient
        """
        return grad_output, None, None, None, None, None


class PerChSTEBase_PTnative(torch.autograd.Function):
    """Base class for customized forward/backward functions.
    There's a family of non-learnable quantizers, such as SAWB, MinMax,
    whose forward can leverage PT native functions and backward is simply STE.
    We just need to calculate scale in the upper level quantizer class then those quantizers
    could all be using the same base "STE function"

    math should be consistent with pytorch: https://pytorch.org/docs/stable/quantization.html
        x_int = round(x/scale + zp)
        x_dq  = (x_int - zp) * scale

    This type of class will be used by Quantizer.forward(), e.g.
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.FloatTensor,
        num_bits: torch.IntTensor,
        clip_valn: torch.FloatTensor,
        clip_val: torch.FloatTensor,
        dequantize: bool = True,
        symmetric: bool = False,
        qlevel_lowering: bool = False,
    ):
        """
        General forward method:
            Set clip values to dtype of input_tensor tensor
            Compute # of quantized levels, scale, and zero point
            Perform PTnative linear quantization on input_tensor tensor
            return output

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            input_tensor_tensor (torch.FloatTensor): Tensor to be quantized.
            num_bits (torch.IntTensor): Number of bit for quantization.
            clip_valn (torch.FloatTensor): Lower clip value bound.
            clip_val (torch.FloatTensor): Upper clip value bound.
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """
        clip_valn, clip_val = transform_clips(
            input_tensor.dtype,
            clip_valn,
            clip_val,
        )
        (
            _,
            scale,
            zero_point,
            qint_l,
            qint_h,
            qint_dtype,
        ) = PerChSTEBase_PTnative.calc_qparams(
            num_bits, clip_valn, clip_val, symmetric, qlevel_lowering
        )
        output = PerChSTEBase_PTnative.linear_quantization(
            input_tensor, scale, zero_point, qint_l, qint_h, qint_dtype, dequantize
        )
        return output

    @classmethod
    def calc_qparams(
        cls,
        num_bits: torch.IntTensor,
        clip_valn: torch.FloatTensor,
        clip_val: torch.FloatTensor,
        symmetric: bool = False,
        qlevel_lowering: bool = False,
    ) -> List[torch.IntTensor, torch.FloatTensor, torch.IntTensor, int, int]:
        """
        Compute the scale and zero_point from num_bits and clip values.
        Also, compute qint bounds for PT clamping.

        Args:
            num_bits (torch.IntTensor): Number of bit for quantization.
            clip_valn (torch.FloatTensor): Lower clip value.
            clip_val (torch.FloatTensor): Upper clip value.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.

        Returns:
            [torch.IntTensor, torch.FloatTensor, torch.IntTensor]: Quantized parameters
        """
        n_levels = 2**num_bits - 2 if qlevel_lowering else 2**num_bits - 1
        scale = (clip_val - clip_valn) / n_levels
        zero_point = (
            torch.zeros_like(scale)
            if symmetric
            else torch.round(-clip_valn / scale).to(torch.int)
        )
        qint_l, qint_h, qint_dtype = PerChSTEBase_PTnative.qint_bounds(
            num_bits, zero_point, symmetric, qlevel_lowering
        )
        return n_levels, scale, zero_point, qint_l, qint_h, qint_dtype

    @classmethod
    def qint_bounds(
        cls,
        num_bits: torch.IntTensor,
        zero_point: torch.IntTensor,
        symmetric: bool = False,
        qlevel_lowering: bool = True,
    ) -> List[int, int, torch.dtype]:
        """
        qlevel_symmetric: shift qlevel from [-2**(b-1), 2**(b-1)-1] to [-2**(b-1)+1, 2**(b-1)-1]
        For int8: [-127,127] ; For int4 [-7,7]
        qint bounds must be ints, not tensors
        """
        num_bits_int = (
            num_bits.item() if isinstance(num_bits, torch.Tensor) else num_bits
        )
        if symmetric and zero_point == 0:
            qlevel_symmetric = 1 if qlevel_lowering else 0
            qint_l, qint_h = (
                -(2 ** (num_bits_int - 1)) + qlevel_symmetric,
                2 ** (num_bits_int - 1) - 1,
            )
            qint_dtype = torch.qint8
        else:  # single_sided or zero_point != 0
            qint_l, qint_h = 0, 2**num_bits_int - 1
            qint_dtype = torch.quint8
        return qint_l, qint_h, qint_dtype

    @classmethod
    def linear_quantization(
        cls,
        input_tensor: torch.FloatTensor,
        scale: torch.FloatTensor,
        zero_point: torch.IntTensor,
        qint_l: int,
        qint_h: int,
        qint_dtype: torch.dtype,
        dequantize: bool = True,
    ) -> torch.Tensor:
        """
        Linear quantization for PTnative STE

        Args:
            input_tensor (torch.FloatTensor): Tensor to be quantized
            scale (torch.FloatTensor): Quantized bin ranges per channel.
            zero_point (torch.IntTensor): Quantized integer bin mapping to fp 0.0 per channel.
            qint_l (int): Quantized integer lower clip value.
            qint_h (int): Quantized integer upper clip value.
            qint_dtype (torch.dtype): Quantized integer dtype.
            dequantize (bool, optional): Specify to return fp or quantized int. Defaults to True.

        Returns:
            torch.Tensor: PTnative quantized or dequantized tensor.
        """
        if dequantize:
            # Note: scale + zero_point are per channel tensors
            output = torch.fake_quantize_per_channel_affine(
                input_tensor.float(),
                scale.float(),
                zero_point,
                axis=0,
                quant_min=qint_l,
                quant_max=qint_h,
            ).to(input_tensor.dtype)
        else:
            # Note: scale is multi-valued, but zero_point isn't...
            output = (
                torch.quantize_per_channel(
                    input_tensor.float(),
                    scale.float(),
                    zero_point,
                    axis=0,
                    dtype=qint_dtype
                )
                .int_repr()
                .clamp(qint_l, qint_h)
            )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        General STE backward method:
            Return grad_output + None args to match forward input_tensor

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            grad_output (torch.FloatTensor): Gradient tensor

        Returns:
            [torch.FloatTensor, None,...,None]: STE Gradient
        """
        return grad_output, None, None, None, None, None, None
