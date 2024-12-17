import pytest
import torch

from fms_mo.aiu_addons.gptq.gptq_aiu_op import register_aiu_gptq_op


input_sizes = [
    {
        "bs": 4,
        "seq_len": 32,
        "hid_dim": 768,
        "out_feat": 3072,
        "n_grp": 6,
    },
]


@pytest.fixture(params=input_sizes)
def get_gptq_gemm_inputs(request):
    sizes = request.param
    compression_factor = 8  # = assume 4-bits compression

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
        (sizes["n_grp"], sizes["out_feat"] // 8), dtype = torch.int32
    )
    scales = torch.randn(
        (sizes["n_grp"], sizes["out_feat"]), dtype=torch.float16,
    )
    g_idx = torch.zeros(sizes["hid_dim"], dtype=torch.int32)

    return (x, qweight, qzeros, scales, g_idx)


def test_gptq_registration() -> None:
    """Call the registration function of GPTQ W4A16 operation, to add it.
    Note: registration must be called before other GPTQ tests.
    """

    register_aiu_gptq_op()
    assert hasattr(torch.ops, "gptq_gemm")
    assert hasattr(torch.ops.gptq_gemm, "i4f16_fxinputs_aiu")
    return


def test_gptq_op(get_gptq_gemm_inputs) -> None:
    """Validate output shapes of GPTQ W4A16 tensors.
    Note: this AIU-compatible operation only returns a zero tensor of the
    expected shape, it does not perform a real W4A16 matmul operation.
    """

    x, qweight, qzeros, scales, g_idx = get_gptq_gemm_inputs
    out = torch.ops.gptq_gemm.i4f16_fxinputs_aiu(x, qweight, qzeros, scales, g_idx)
    assert out.size() == torch.Size((x.size()[:-1] + (qweight.size(0),)))
