# Third Party

# from mx import Linear as Linear_mx  # Need to amend mx's Linear class
# Third Party
from mx.elemwise_ops import quantize_elemwise_op
from mx.linear import linear
from mx.specs import apply_mx_specs, mx_assert_test
import numpy as np
import torch
import torch.nn.functional as F


class LinearMX(torch.nn.Linear):  # amend init and repr
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        mx_specs=None,
        name=None,
        **kwargs,  # [CL] current qmodel_prep will pass lots of stuff in
    ):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None

        self.name = name
        self.prequantized_weights = False
        self.mx_specs = apply_mx_specs(mx_specs)
        super().__init__(
            in_features, out_features, bias, device=kwargs.get("device", "cuda")
        )  # [CL] would like to pass device to nn.Linear

    def apply_mx_specs(self, mx_specs):
        mx_assert_test(mx_specs)
        self.mx_none = mx_specs is None
        self.mx_specs = apply_mx_specs(mx_specs)

    def append_name(self, postfix):
        self.name += postfix

    def prequantize_weights(self):
        # Can't prequantize if not using bfloat weights
        if self.mx_none:
            return

        assert (
            self.mx_specs["round"] == "even"
        ), "Bfloat round should be 'even' for prequantizing weights."
        assert (
            torch.cuda.is_bf16_supported()
        ), "Current device does not support bfloat16"
        assert self.mx_specs[
            "bfloat_subnorms"
        ], "Bfloat_subnorms should be set to True for prequantizing weights."
        assert (
            self.mx_specs["bfloat"] == 16
        ), "Only Bfloat16 is supported for prequantizing weights."

        with torch.no_grad():
            self.weight.data = quantize_elemwise_op(
                self.weight.data,
                mx_specs=self.mx_specs,
                round=self.mx_specs["round_weight"],
            ).to(torch.bfloat16)

            if self.bias is not None:
                self.bias.data = quantize_elemwise_op(
                    self.bias.data,
                    mx_specs=self.mx_specs,
                    round=self.mx_specs["round_weight"],
                ).to(torch.bfloat16)

        self.prequantized_weights = True

    def forward(self, inputs):
        if self.mx_none:
            return super().forward(inputs)

        if self.prequantized_weights:
            assert not self.training, "Cannot use prequantized weights when training!"

        return linear(
            input=inputs,
            weight=self.weight,
            bias=self.bias,
            mx_specs=self.mx_specs,
            prequantized_weights=self.prequantized_weights,
            name=self.name,
        )

    def extra_repr(self) -> str:
        """[CL] to make it clear it's MX Linear"""
        repr_str = (
            f"in={self.in_features},out={self.out_features},"
            f"w_fmt={self.mx_specs['w_elem_format']},a_fmt={self.mx_specs['a_elem_format']}"
            f"blk_size={self.mx_specs['block_size']}"
        )
        return repr_str


class ResidualMLP(torch.nn.Module):
    def __init__(self, hidden_size, device="cuda"):
        super(ResidualMLP, self).__init__()

        self.layernorm = torch.nn.LayerNorm(hidden_size, device=device)
        self.dense_4h = torch.nn.Linear(hidden_size, 4 * hidden_size, device=device)
        self.dense_h = torch.nn.Linear(4 * hidden_size, hidden_size, device=device)
        self.dummy = torch.nn.Linear(hidden_size, hidden_size, device=device)
        # add a dummy layer because by default we skip 1st/last, if there are only 2 layers, all will be skipped

    def forward(self, inputs):
        norm_outputs = self.layernorm(inputs)

        # MLP
        proj_outputs = self.dense_4h(norm_outputs)
        proj_outputs = F.gelu(proj_outputs)
        mlp_outputs = self.dense_h(proj_outputs)
        mlp_outputs = self.dummy(mlp_outputs)

        # Residual Connection
        outputs = inputs + mlp_outputs

        return outputs


if __name__ == "__main__":
    # Add config arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--hidden_size", default=128)
    # parser.add_argument("--device", default='cuda')
    # args = parser.parse_args()
    # Standard
    from functools import partial

    # Third Party
    from mx import MxSpecs

    # Local
    from fms_mo import qconfig_init, qmodel_prep

    x = np.random.randn(16, 128)
    x = torch.tensor(x, dtype=torch.float32, device="cuda")

    # --- Test 0. Run MLP as is
    mlp = ResidualMLP(128)
    # mlp.to("cuda")
    with torch.no_grad():
        out = mlp(x)
    print(mlp)

    # --- Test 1. fms-mo qmodel_prep, replace Linear with our QLinear
    qcfg = qconfig_init()
    qcfg["nbits_a"] = 8
    qcfg["nbits_w"] = 8
    model = qmodel_prep(mlp, x, qcfg)
    with torch.no_grad():
        out8 = model(x)
    print(model)

    # --- Test 2. now change mapping to MX
    # NOTE this is what will happen under the hood when we update qmodel_prep() in the near future
    #       it's just an explicit test for now
    mx_specs = MxSpecs()
    mx_specs["scale_bits"] = 8
    mx_specs["w_elem_format"] = "fp4_e2m1"
    mx_specs["a_elem_format"] = "fp4_e2m1"
    mx_specs["block_size"] = 32
    mx_specs["bfloat"] = 16
    mx_specs["custom_cuda"] = True
    qcfg["mx_specs"] = mx_specs
    mlp = ResidualMLP(128)  # fresh model
    MXLinear = partial(LinearMX, mx_specs=qcfg["mx_specs"])
    qcfg["mapping"] = {
        torch.nn.Linear: {
            "from": torch.nn.Linear,
            "to": MXLinear,
            "otherwise": MXLinear,
        },
    }
    model = qmodel_prep(mlp, x, qcfg)
    with torch.no_grad():
        out4 = model(x)
    print(model)

    # --- final output
    print(f"ref output", out)
    print(f"int8 output", out8)
    print(f"mxfp4 output", out4)
    print("DONE!")
