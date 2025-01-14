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
Qmax Quantizers and STEs
"""

from typing import Tuple

# Third Party
import torch

# Local
from fms_mo.quant_refactor.base_quant import QuantizerBase, Qscheme
from fms_mo.quant_refactor.base_tensor import (
    PerTensorSTEBase_PTnative,
    PerTensorSTEQmax,
    PerTensorSTEQmax_PTnative,
)

perTQscheme_default = Qscheme(
    unit="perT",
    symmetric=True,
    Nch=None,
    Ngrp=None,
    single_sided=True,
    qlevel_lowering=False,
)
clip_valn_default = torch.tensor(-8.0)
clip_val_default = torch.tensor(8.0)

class Qmax_new(QuantizerBase):
    """
    SAWB with custom backward (gradient pass through for clip function)
    if align_zero: quantizer = SAWBSTE() for coded sawb such as 103, 403, 803
    if not align_zero: quantizer = SAWBZeroSTE() for normal precision setting such as 2, 4, 8
    Qmax can quantize both weights and activations

    Extends:
        QuantizerBase
    """

    def __init__(
        self,
        num_bits: torch.IntTensor,
        init_clip_valn: torch.FloatTensor = clip_valn_default,
        init_clip_val: torch.FloatTensor = clip_val_default,
        qscheme: Qscheme = perTQscheme_default,
        dequantize: bool = True,
        align_zero: bool = False,
        clipSTE: bool = True,
        minmax: bool = False,
        extend_act_range: bool = False,
        **kwargs,
    ):
        """
        Init SAWB Quantizer

        Args:
            num_bits (torch.IntTensor): Number of bits for quantization.
            init_clip_valn (torch.FloatTensor, optional): Lower clip value bound. Defaults to -8.0.
            init_clip_val (torch.FloatTensor, optional): Upper clip value bound. Defaults to 8.0.
            qscheme (Qscheme, optional): Quantization scheme.
                Defaults to Qscheme( unit="perT", symmetric=True, Nch=None, Ngrp=None,
                                       single_sided=False, qlevel_lowering=False, ).
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            align_zero (bool, optional): Specify using qlevel_lowering. Defaults to False.
            clipSTE (bool, optional): Specify to not clip backward() input tensor.
                Defaults to False.
            extend_act_range (bool, optional): Specify using full quantization range.
                Defaults to False.
            use16bins (bool, optional): Specify using special 16bin quantization. Defaults to False.
        """
        super().__init__(
            num_bits,
            dequantize,
            qscheme=qscheme,
            use_PT_native_Qfunc=kwargs.get("use_PT_native_Qfunc", False),
        )

        if not self.training:
            with torch.no_grad():
                self.clip_valn.data *= init_clip_valn
                self.clip_val.data *= init_clip_val

        self.align_zero = align_zero
        self.clipSTE = clipSTE
        self.minmax = minmax
        self.extend_act_range = extend_act_range

        # Training
        self.movAvgFac = 0.1
        self.Niter = 0
        # copy new clip vals in forward without training from nn.Module
        self.recompute_clips = False

        self.set_quantizer()

    def copy_legacy_vars(self):
        """
        Copy legacy variables to newer versions.
        Allows for better backwards compatability
        """
        self.qscheme.qlevel_lowering = self.align_zero

    def set_quantizer(self):
        """
        Set quantizer STE based on current member variables
        """
        self.copy_legacy_vars()  # copy align_zero

        if self.use_PT_native_Qfunc:
            if self.extend_act_range:
                self.quantizer_name = "QmaxExtend"
                self.set_clip_ratio()
                self.quantizer = QmaxExtendRangeSTE_PTnative
            else:
                self.quantizer_name = "Qminmax" if self.minmax else "Qmax"
                self.quantizer = PerTensorSTEQmax_PTnative
        else:
            if self.minmax:
                self.quantizer_name = "Qminmax"
                if self.perGrp:
                    self.quantizer = QminmaxPerGpSTE_new
                else:
                    self.quantizer = (
                        QminmaxPerChSTE_new if self.perCh else QminmaxSTE_new
                    )
            elif self.extend_act_range:
                self.quantizer = QmaxExtendRangeSTE_new
                self.quantizer_name = "QmaxExtend"
                self.set_clip_ratio()
            else:
                self.quantizer_name = "Qmax"
                if self.perGrp:
                    self.quantizer = QmaxPerGpSTE_new
                else:
                    self.quantizer = QmaxPerChSTE_new if self.perCh else QmaxSTE_new

    def set_clip_ratio(self):
        """
        Compute clip_ratio used in Extended Range STEs
        """
        n_half = 2 ** (self.num_bits - 1)
        self.clip_ratio = -n_half / (n_half - 1)

    def forward(self, input_tensor: torch.FloatTensor) -> torch.Tensor:
        """
        Qmax forward() function.
        Has codepath for saving previous runs computed clip values.

        Args:
            input_tensor (torch.FloatTensor): Tensor to be quantized.

        Returns:
            torch.Tensor: Quantized or dequantized tensor.
        """
        if self.perCh:
            if self.minmax:
                clip_val_new = torch.max(
                    input_tensor.reshape([self.perCh, -1]), dim=1
                ).values
                clip_valn_new = torch.min(
                    input_tensor.reshape([self.perCh, -1]), dim=1
                ).values
            else:
                clip_val_new = torch.max(
                    input_tensor.abs().reshape([self.perCh, -1]), dim=1
                ).values
                clip_valn_new = -clip_val_new
            assert (
                len(clip_val_new) == input_tensor.shape[0]
            ), f"dimension error, input{input_tensor.shape}, clipval{clip_val_new.shape}"
        elif self.perGrp:
            if self.minmax:
                clip_val_new = torch.max(
                    input_tensor.reshape(self.perGrp), dim=1
                ).values
                clip_valn_new = torch.min(
                    input_tensor.reshape(self.perGrp), dim=1
                ).values
            else:
                clip_val_new = torch.max(
                    input_tensor.abs().reshape(self.perGrp), dim=1
                ).values
                clip_valn_new = -clip_val_new
            assert len(clip_val_new) == (
                input_tensor.shape[0] * input_tensor.shape[1] // self.perGrp[1]
            ), f"dimension error, input{input_tensor.shape}, clip_val{clip_val_new.shape}"
        elif self.extend_act_range:
            if input_tensor.max() >= input_tensor.min().abs():
                clip_val_new = input_tensor.max()
                clip_valn_new = clip_val_new * self.clip_ratio
            else:
                clip_valn_new = input_tensor.min()
                clip_val_new = clip_valn_new / self.clip_ratio
        else:
            if self.minmax:  # asymmetric
                clip_val_new = input_tensor.max()
                clip_valn_new = input_tensor.min()
            else:  # symmetric
                clip_val_new = input_tensor.abs().max()
                clip_valn_new = -clip_val_new

        if len(clip_val_new.shape) == 0:
            clip_val_new = clip_val_new.unsqueeze(dim=0)
        if len(clip_valn_new.shape) == 0:
            clip_valn_new = clip_valn_new.unsqueeze(dim=0)

        if (self.Niter == 0 and self.training) or self.recompute_clips:
            # to avoid unintended bwd ops added to the graph, cause memory leak sometimes
            with torch.no_grad():
                self.clip_val.copy_(clip_val_new)
                self.clip_valn.copy_(clip_valn_new)

        if self.training:

            output = self.quantizer.apply(
                input_tensor,
                self.num_bits,
                clip_valn_new,
                clip_val_new,  # use new clip_vals first, then do moving average
                self.dequantize,
                self.qscheme.symmetric,
                self.qscheme.qlevel_lowering,
                self.minmax,
            )
            with torch.no_grad():
                # to avoid unintended bwd ops added to the graph, cause memory leak sometimes
                self.clip_val.copy_(
                    self.clip_val * (1.0 - self.movAvgFac)
                    + clip_val_new * self.movAvgFac
                )
                if self.extend_act_range:
                    self.clip_valn.copy_(self.clip_val * self.clip_ratio)
                else:
                    self.clip_valn.copy_(
                        self.clip_valn * (1.0 - self.movAvgFac)
                        + clip_valn_new * self.movAvgFac
                    )
        else:
            output = self.quantizer.apply(
                input_tensor,
                self.num_bits,
                self.clip_valn,
                self.clip_val,
                self.dequantize,
                self.qscheme.symmetric,
                self.qscheme.qlevel_lowering,
                self.minmax,
            )

        self.Niter += 1
        return output


class QmaxSTE_new(PerTensorSTEQmax):
    """
    QMax with zero alignment (symmetric)

    Extends:
        PerTensorSTEQmax: Uses PerTensorSTEQmax.forward()
    """

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for Qmax STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.FloatTensor): Gradient to clip

        Returns:
            [torch.FloatTensor, None,...,None]: Gradients
        """
        return grad_output, None, None, None, None, None, None, None


class QminmaxSTE_new(PerTensorSTEQmax):
    """minmax with zero alignment (asymmetric)
    Dequantization always enabled (cannot be turned off, at this time)
    """

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for Qminmax STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.FloatTensor): Gradient to clip

        Returns:
            [torch.FloatTensor, None,...,None]: Gradients
        """
        return grad_output, None, None, None, None, None, None, None


class QmaxExtendRangeSTE_new(torch.autograd.Function):
    """
    2-sided Max quantizer using a single clip

    Assume negative clip (clip_valn) was derived from positive clip (clip_val)
    using extended range, such that:
        clip_valn / clip_val = -2^(b-1)/(2^(b-1)-1)
    Example:
        b = 8 --> [-128/127 * clip_val, clip_val]
        b = 4 --> [-8/7 * clip_val, clip_val]

    This quantizer has NO zero_point, and it is therefore suitable for BMM
    Zero alignment (FP=0 --> INT=0 --> FP=0) is guaranteed (align_zero is ignored)
    No in_place functionalities
    Dequantization is functional
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.FloatTensor,
        num_bits: torch.IntTensor,
        clip_valn: torch.FloatTensor,
        clip_val: torch.FloatTensor,
        dequantize: bool = True,
        _symmetric: bool = False,
        _qlevel_lowering: bool = True,
        _use_minmax: bool = False,
    ):
        """
        Forward function for QmaxExtendRangeSTE

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            input_tensor (torch.FloatTensor): Tensor to be quantized.
            num_bits (torch.IntTensor): Number of bit for quantization.
            clip_valn (torch.FloatTensor): Lower clip value bound.
            clip_val (torch.FloatTensor): Upper clip value bound.
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.
            _use_minmax (bool, optional): Specify using Qminmax. Defaults to False.

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """
        clip_val, clip_valn = clip_val.to(input_tensor.dtype), clip_valn.to(
            input_tensor.dtype
        )

        scale = clip_val / (2 ** (num_bits - 1) - 1)
        qint_min, qint_max = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1

        # quantize to range [-2^(b-1), 2^(b-1)-1]
        output = torch.round(input_tensor / scale)
        output = output.clamp(qint_min, qint_max)
        if dequantize:
            output = output * scale
        else:
            output = output.to(torch.int8)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for Qmax Extended Range STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.FloatTensor): Gradient to clip

        Returns:
            [torch.FloatTensor, None,...,None]: Gradients
        """
        return grad_output, None, None, None, None, None, None, None


class QmaxExtendRangeSTE_PTnative(PerTensorSTEQmax_PTnative):
    """
    Qmax Extended Range STE w/ PT native kernel

    Extends:
        PerTensorSTEQmax_PTnative
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
        qlevel_lowering: bool = True,
        use_minmax: bool = False,
    ) -> torch.Tensor:
        """ "
        Forward function for QmaxExtendRangeSTE w/ PTnative kernel

        Args:
            ctx (torch.autograd.Function): Forward/Backward context object.
            input_tensor (torch.FloatTensor): Tensor to be quantized.
            num_bits (torch.IntTensor): Number of bit for quantization.
            clip_valn (torch.FloatTensor): Lower clip value bound.
            clip_val (torch.FloatTensor): Upper clip value bound.
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.
            use_minmax (bool, optional): Specify using Qminmax. Defaults to False.

        Returns:
            torch.Tensor: Dequantized or Quantized output tensor.
        """
        (
            _,
            _,
            scale,
            zero_point,
            qint_l,
            qint_h,
            qint_dtype,
        ) = QmaxExtendRangeSTE_PTnative.calc_qparams(
            num_bits, clip_valn, clip_val, symmetric, qlevel_lowering, use_minmax
        )
        output = PerTensorSTEBase_PTnative.linear_quantization(
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
        qlevel_lowering: bool = True,
        use_minmax: bool = False,
    ) -> Tuple[
        torch.IntTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.IntTensor,
        int,
        int,
        torch.dtype,
    ]:
        """
        Compute the scale and zero_point from num_bits and clip values

        Args:
            num_bits (torch.IntTensor): Number of bit for quantization.
            clip_valn (torch.FloatTensor): Lower clip value.
            clip_val (torch.FloatTensor): Upper clip value.
            symmetric (bool, optional): Specify if clip values are symmetric. Defaults to False.
            qlevel_lowering (bool, optional): Specify lowering of quantized levels.
                Defaults to True.
            use_minmax (bool, optional): Specify using Qminmax. Defaults to False.

        Returns:
            [torch.IntTensor, torch.FloatTensor, torch.IntTensor
            torch.IntTensor, torch.IntTensor, torch.IntTensor, torch.dtype]:
                Quantized parameters for PTnative kernels
        """
        n_levels = 2**num_bits - 2
        scale = 2 * clip_val / n_levels
        zero_point = torch.tensor(0)
        qint_min, qint_max = -(2 ** (num_bits - 1)), 2 ** (num_bits - 1) - 1
        qint_dtype = torch.qint8

        return n_levels, clip_val, scale, zero_point, qint_min, qint_max, qint_dtype


# Placeholder classes for PerCh/PerGp - need to rework #
class QmaxPerChSTE_new(torch.autograd.Function):
    """
    Max with zero alignment (symmetric)
    "dequantize=False" option is functional
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor,
        num_bits,
        _dequantize,
        inplace,
        _cvn,
        cv,
        align_zero,
    ) -> torch.FloatTensor:
        """
        TODO (bmgroth): docstring
        """
        if inplace:
            ctx.mark_dirty(input)
        scale = (2**num_bits - 2) if align_zero else (2**num_bits - 1)
        zero_point = 0.0
        _clip_val = cv
        # here use symmetric similar to sawbperCh code
        _nspace = 2**num_bits - 2  # lose one level
        int_l = -(2 ** (num_bits - 1)) + 1
        int_u = -int_l  # symmetric
        scale = (
            cv * 2 / (2**num_bits - 2)
        )  # original SAWB assumes odd number of bins when calc clip_val
        zero_point = torch.zeros_like(scale)  # centers around 0 and align 0
        # FIXME, fake quantize function only support float.
        output = torch.fake_quantize_per_channel_affine(
            input_tensor.float(),
            scale.float(),
            zero_point.float(),
            axis=0,
            quant_min=int_l,
            quant_max=int_u,
        ).to(input_tensor.dtype)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for Qmax Per Channel STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.FloatTensor): Gradient to clip

        Returns:
            [torch.FloatTensor, None,...,None]: Gradients
        """
        return grad_output, None, None, None, None, None, None


class QmaxPerGpSTE_new(torch.autograd.Function):
    """
    Max with zero alignment (symmetric)
    per group quantization
    """

    @staticmethod
    def forward(
        ctx, input_tensor, num_bits, _dequantize, inplace, cv, _cvn, align_zero
    ) -> torch.FloatTensor:
        """
        TODO (bmgroth): docstring
        """
        if inplace:
            ctx.mark_dirty(input_tensor)
        _clip_val = cv
        input_shape = input_tensor.shape
        clip_val_shape = cv.shape
        # use clip_val shape to reshape input
        input_tensor = input_tensor.reshape(clip_val_shape[0], -1)
        scale = (2**num_bits - 2) if align_zero else (2**num_bits - 1)
        zero_point = 0.0

        # here use symmetric similar to sawbperCh code
        _nspace = 2**num_bits - 2  # lose one level
        int_l = -(2 ** (num_bits - 1)) + 1
        int_u = -int_l  # symmetric
        scale = (
            cv * 2 / (2**num_bits - 2)
        )  # original SAWB assumes odd number of bins when calc clip_val
        zero_point = torch.zeros_like(scale)  # centers around 0 and align 0
        # FIXME, fake quantize function only support float.
        output = torch.fake_quantize_per_channel_affine(
            input_tensor.float(),
            scale.float(),
            zero_point.float(),
            axis=0,
            quant_min=int_l,
            quant_max=int_u,
        ).to(input_tensor.dtype)
        # reshape back to original shape
        output = output.reshape(input_shape)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for Qmax Per Group STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.FloatTensor): Gradient to clip

        Returns:
            [torch.FloatTensor, None,...,None]: Gradients
        """
        return grad_output, None, None, None, None, None, None


class QminmaxPerChSTE_new(torch.autograd.Function):
    """
    per channel minmax with zero alignment (asymmetric)
    """

    @staticmethod
    def forward(
        ctx, input_tensor, num_bits, _dequantize, inplace, cv, cvn, align_zero
    ) -> torch.FloatTensor:
        """
        TODO (bmgroth): docstring
        """
        if inplace:
            ctx.mark_dirty(input_tensor)
        cv, cvn = cv.to(input_tensor.dtype), cvn.to(input_tensor.dtype)
        scale = (2**num_bits - 1) / (cv - cvn)
        zero_point = cvn * scale
        if align_zero:
            zero_point = torch.round(zero_point)
        output = (input_tensor.clamp(cvn[:, None], cv[:, None]) - cvn[:, None]) * scale[
            :, None
        ]
        output = (torch.round(output) + zero_point[:, None]) / scale[:, None]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for Qminmax Per Channel STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.FloatTensor): Gradient to clip

        Returns:
            [torch.FloatTensor, None,...,None]: Gradients
        """
        return grad_output, None, None, None, None, None, None


class QminmaxPerGpSTE_new(torch.autograd.Function):
    """
    per group minmax with zero alignment (asymmetric)
    """

    @staticmethod
    def forward(
        ctx, input_tensor, num_bits, _dequantize, inplace, cv, cvn, align_zero
    ) -> torch.FloatTensor:
        """
        TODO (bmgroth): docstring
        """
        if inplace:
            ctx.mark_dirty(input_tensor)
        cv, cvn = cv.to(input_tensor.dtype), cvn.to(input_tensor.dtype)
        input_shape = input_tensor.shape
        clip_val_shape = cv.shape
        input_tensor = input_tensor.reshape(clip_val_shape[0], -1)
        scale = (2**num_bits - 1) / (cv - cvn)
        zero_point = cvn * scale
        if align_zero:
            zero_point = torch.round(zero_point)
        output = (input_tensor.clamp(cvn[:, None], cv[:, None]) - cvn[:, None]) * scale[
            :, None
        ]
        output = (torch.round(output) + zero_point[:, None]) / scale[:, None]
        # reshape back to original shape
        output = output.reshape(input_shape)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for Qminmax Per Group STE

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.FloatTensor): Gradient to clip

        Returns:
            [torch.FloatTensor, None,...,None]: Gradients
        """
        return grad_output, None, None, None, None, None, None
