# Third Party
# import argparse
# Third Party
from mx import Linear as Linear_mx  # Need to amend mx's Linear class
import numpy as np
import torch
import torch.nn.functional as F


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

    mx_specs = MxSpecs()

    mx_specs["scale_bits"] = 8
    mx_specs["w_elem_format"] = "fp4_e2m1"
    mx_specs["a_elem_format"] = "fp4_e2m1"
    mx_specs["block_size"] = 32
    mx_specs["bfloat"] = 16
    mx_specs["custom_cuda"] = True

    x = np.random.randn(16, 128)
    x = torch.tensor(x, dtype=torch.float32, device="cuda")

    # Test 0. Run MLP as is
    mlp = ResidualMLP(128)
    # mlp.to("cuda")
    with torch.no_grad():
        out = mlp(x)
    print(mlp)

    # --- fms-mo starts here
    qcfg = qconfig_init()
    qcfg["nbits_a"] = 8
    qcfg["nbits_w"] = 8
    # Test 1. normal qmodel_prep will replace Linear with our QLinear
    model = qmodel_prep(mlp, x, qcfg)
    with torch.no_grad():
        out8 = model(x)
    print(model)

    # Test 2. now change mapping
    # NOTE this is what will happen under the hood when we update qmodel_prep() in the near future
    #       it's just an explicit test for now
    qcfg["mx_specs"] = mx_specs
    mlp = ResidualMLP(128)  # fresh model
    MXLinear = partial(Linear_mx, mx_specs=qcfg["mx_specs"])
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

    print(f"ref output", out)
    print(f"int8 output", out8)
    print(f"mxfp4 output", out4)
    print("DONE!")
