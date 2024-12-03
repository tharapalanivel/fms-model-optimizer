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
PACT2Symmetric Quantizer
"""

# Third Party
import torch

# Local
from fms_mo.quant_refactor.base_quant import QuantizerBase, sqQscheme
from fms_mo.quant_refactor.base_tensor import (
    PerTensorSTEBase,
    PerTensorSTEBase_PTnative,
)


class PACT2Sym_new(QuantizerBase):
    """
    Two-sided PACT with symmetric clip values

    Extends:
        QuantizerBase
    """

    def __init__(
        self,
        num_bits: torch.IntTensor,
        init_clip_valn: torch.FloatTensor = torch.tensor(-8.0),
        init_clip_val: torch.FloatTensor = torch.tensor(8.0),
        qscheme=sqQscheme(
            unit="perT",
            symmetric=True,
            Ngrp_or_ch=None,
            single_sided=False,
            qlevel_lowering=False,
        ),
        dequantize: bool = True,
        **kwargs
    ):
        """
        Init PACT2Sym quantizer

        Args:
            num_bits (torch.IntTensor): Number of bits for quantization.
            init_clip_valn (torch.FloatTensor, optional): Lower clip value bound. Defaults to -8.0.
            init_clip_val (torch.FloatTensor, optional): Upper clip value bound. Defaults to 8.0.
            qscheme (sqQscheme, optional): Quantization scheme.
                Defaults to sqQscheme( unit="perT", symmetric=False, Ngrp_or_ch=None,
                                       single_sided=False, qlevel_lowering=False, ).
            dequantize (bool, optional): Return dequantized or int tensor. Defaults to True.
            kwargs.use_PT_native_Qfunc (bool, optional): Use native PT quantizer.
                Defaults to False.
        """
        super().__init__(
            num_bits,
            dequantize,
            qscheme=qscheme,
            use_PT_native_Qfunc=kwargs.get("use_PT_native_Qfunc", False),
        )

        with torch.no_grad():
            self.clip_valn.data *= init_clip_valn
            self.clip_val.data *= init_clip_val

        self.set_quantizer()

    def set_quantizer(self):
        """
        Set quantizer STE based on current member variables
        """
        if self.use_PT_native_Qfunc:
            self.quantizer = PerTensorSTEBase_PTnative
        else:
            self.quantizer = PACT2Sym_STE_new


class PACT2Sym_STE_new(PerTensorSTEBase):
    """
    Symmetric with zero in the center. For example, 4bit -- > [-7, 7] with FP0 align to INT0

    Extends:
        PerTensorSTEBase: Uses PerTensorSTEBase.forward()
    """

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for PACT2Sym

        Args:
            ctx (torch.autograd.Function): Context object.
            grad_output (torch.FloatTensor): Gradient to clip

        Returns:
            [torch.FloatTensor, torch.FloatTensor, None,...,None]: Gradients
        """
        input_tensor, _, _, clip_val, _, _ = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = torch.where(
            input_tensor <= -clip_val, torch.zeros_like(grad_input), grad_input
        )
        grad_input = torch.where(
            input_tensor >= clip_val, torch.zeros_like(grad_input), grad_input
        )

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(
            torch.logical_and(input_tensor < clip_val, input_tensor > -clip_val),
            torch.zeros_like(grad_alpha),
            grad_alpha,
        )
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        return grad_input, grad_alpha, None, None, None, None, None
