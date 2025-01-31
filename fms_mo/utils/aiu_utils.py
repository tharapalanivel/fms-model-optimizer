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

"""Functions for deploying quantized models to AIU"""

# Standard
import json
import logging
import warnings

# Third Party
from packaging.version import Version
from transformers.tokenization_utils_base import BatchEncoding
import torch

# Local
from fms_mo.fx.utils import lname_to_org_name, plot_graph_module
from fms_mo.utils.utils import default_device_selection

disable_torchtrt = True
logger = logging.getLogger(__name__)

DEFAULT_DEVICE = default_device_selection()


def add_aiu_suffix(node_name: str) -> str:
    """Modify name to satisfy AIU requirements for addmm nodes.
    Only addmm require suffix, not mm without bias.
    """
    if "addmm" in node_name:
        node_name += "_MatMul"
    return node_name


def cus_backend_generate_preccfg(
    gm_fx,
    sample_inp,
    LUTorg_mod_clipvals=None,
    LUTquant_mods=None,
    LUTmod2org_name=None,
    enforce_clipval_ratio=False,
    comp_lay0_by_dropout=False,
    output_dir=None,
    exportSVG=False,
    mappables=None,
    sym_act=None,
):
    """This func is meant to be used as a custom backend for a "traceable" Qmodel
    e.g., with torch.no_grad():
              model_compiled = torch.compile(model, backend=[this])
              model_compiled(**example_input)

    Args:
        comp_lay0_by_dropout: (bool), need to pass dropout prob if need compensation
        sym_act: (list), this is a list of original module names that is using sym activations.

    resulting json files should be saved in the same folder
    NOTE:
    1. to make a Qmodel traceable, we need to swap QLinear with QLinearINT8Deploy
    2. QConv2d still needs implement an equivalant QConv_deploy wrapper
    3. the key component here is:
        a) use meta data in aten nodes to identify the original module
        b) then use the LUTorg_mod_clipvals to find the corresponding clipvals in that module
    4. Newer PT alters the Linear instance, so we can no long assume module under FX GM is the same
        as the original module.
        For example, a given mod in FX gm may still look like Linear, but
            id(mod) != id(original_mod_before_dynamo)
        Better to use id(mod.weight) to match, or make a guess from Lname

    """
    # Third Party
    from torch._decomp import get_decompositions
    from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_compiler

    show_details_onSVG = False
    MAX_NODE_TO_PLOT = 1000
    if exportSVG:
        plot_graph_module(
            gm_fx,
            outputname="FX.svg",
            Nnode_to_plot=MAX_NODE_TO_PLOT,
            show_details=show_details_onSVG,
        )

    LUTfx_name_to_node = {n.name: n for n in gm_fx.graph.nodes}
    global LUTaten_name_to_org_mod  # pylint: disable=global-statement
    LUTaten_name_to_org_mod = {}
    if Version(torch.__version__) < Version("2.1"):
        LUTmod_name_fx2org = {
            n_fx: LUTmod2org_name[m]
            for n_fx, m in gm_fx.named_modules()
            if isinstance(m, mappables)
        }
    else:
        # see NOTE 4
        LUTmod_name_fx2org = {}
        call_mod_nodes = [n for n in gm_fx.graph.nodes if n.op == "call_module"]
        for n in call_mod_nodes:
            fx_mod_name = list(n.meta["nn_module_stack"].keys())[-1]
            org_mod_name, mod_class = list(n.meta["nn_module_stack"].values())[-1]
            if mod_class in mappables:
                LUTmod_name_fx2org[fx_mod_name] = lname_to_org_name(org_mod_name)

    preccfg_json = {
        "precision": {},
        "alpha_ap": {},
        "alpha_an": {},
        "alpha_wp": {},
        "alpha_wn": {},
        "zero_shift": {},
        "W_int": {},
        "input_zp": {},
    }
    # NOTE W_int is just a temp storage, will convert to "zero_shift" later
    QKVsiblings = []
    if enforce_clipval_ratio:
        # user needs to pass desired ratio thru this flag if need adj
        # possible values are [-128/127, -1, False], i.e. False = no change
        assert enforce_clipval_ratio in [
            -128 / 127,
            -1,
        ], (
            f"Unconventional clipval ratio enforcement {enforce_clipval_ratio}, "
            "should be -128/127 or -1"
        )

    def fake_fwd_comp(
        gm_aten,
        inp,  # pylint: disable=unused-argument
    ):
        if exportSVG:
            plot_graph_module(
                gm_aten,
                outputname="Aten.svg",
                Nnode_to_plot=MAX_NODE_TO_PLOT,
                show_details=show_details_onSVG,
            )
        logger.info("Aten GM used for AIU conversion\n")
        # First, find target nodes in Aten GM and do some clean-up, e.g. QKV sync and QBmm check
        # NOTE: qbmm is not in the gm_fx, mapping of bmm Op to QBmm is done in the next part
        # fms_mo.imatmul is a custom Op we register for QLinearINT8Deploy
        aten_node_of_interest = {
            "addmm": [],
            "conv": [],
            "bmm": [],
            "imatmul": [],
            "aten.mm": [],
        }
        QKVbranch_point = []
        for n in gm_aten.graph.nodes:
            for op_kw, node_list in aten_node_of_interest.items():
                if op_kw in str(n.target):
                    op_type = (
                        "linear" if op_kw in ["addmm", "imatmul", "aten.mm"] else op_kw
                    )
                    if Version(torch.__version__) < Version("2.1"):
                        # for PT2.0
                        fx_mod_name = list(n.meta["nn_module_stack"].keys())[-1]
                        org_mod_name = LUTmod_name_fx2org.get(fx_mod_name, None)
                    else:
                        # for newer PT
                        n_fx_org_mod_name = list(n.meta["nn_module_stack"].values())[
                            -1
                        ][0]
                        org_mod_name = lname_to_org_name(n_fx_org_mod_name)
                    LUTaten_name_to_org_mod[n.name] = org_mod_name  # save for later use
                    # make sure the original module is quantized
                    if org_mod_name in LUTquant_mods[op_type]:
                        node_list.append(n)
            # also try to find QKV siblings, identify those nodes with 4 users, i.e. Q+K+V+shortcut
            if len(n.users) == 4:
                QKVbranch_point.append(n)

        # NOTE clean-up node_of_interest for a few cases:
        if not any("QBmm" in k for k in LUTorg_mod_clipvals.keys()):
            # Case a: QBmm was not used in the model -> no clipvals has 'QBmm' in name (LUT.keys())
            #   do not collect and handle bmm in json -> set bmm in node_of_interest to empty
            aten_node_of_interest["bmm"] = []
        # other cases => handled in search already, see Line3177
        # Case b: some or all Linears are not quantized
        # Case c: CNN with linears, some are not quantized

        # identify QKV siblings and make a LUT, only for Linear related Ops/nodes
        def is_any_child_on_the_list(n_curr, most_wanted, search_depth=5):
            curr_search_depth = 0
            while curr_search_depth < search_depth:
                if n_curr in most_wanted:
                    return curr_search_depth, add_aiu_suffix(n_curr.name)
                    # return AIU node name, not node itself
                n_curr = list(n_curr.users)[0]  # no need to recursive in this case
                curr_search_depth += 1
            return curr_search_depth, None

        n_aten_lin = (
            aten_node_of_interest["addmm"]
            + aten_node_of_interest["imatmul"]
            + aten_node_of_interest["aten.mm"]
        )
        for n_br in QKVbranch_point:
            curr_grp_names = []
            curr_grp_depth = []
            for n_cand in n_br.users:  # each branch point has 4 users, .users is a dict
                depth, name = is_any_child_on_the_list(n_cand, n_aten_lin)
                curr_grp_names.append(name)
                curr_grp_depth.append(depth)
            # usually there will be 3 Linear and 1 None (from shortcut, but there's a chance it
            # can find something within the given depth)
            # The 3 linears should have exactly the same 'depth', and exactly 3 of them, not 2 or 4
            curr_grp_depth_count = [
                curr_grp_depth.count(elem_i) for elem_i in curr_grp_depth
            ]
            idx_item_to_remove = [c != 3 for c in curr_grp_depth_count].index(True)
            curr_grp_depth_count.pop(
                idx_item_to_remove
            )  # pop() is an inplace action, the list will be 1 element shorter,
            # but remember the return of this statement is the popped element
            if all(c == 3 for c in curr_grp_depth_count):
                # only when the remaining elements have exactly the same depth, hence
                # "depth count"==3, it's a QKV group
                curr_grp_names.pop(idx_item_to_remove)
                QKVsiblings.append(curr_grp_names)
            else:
                continue

        # --- Second, start to go thru nodes of interests and collect info into json
        # --- case 1: record clipvals/W_int for Linear/addmm/imatmul
        for n_aten in (
            aten_node_of_interest["addmm"]
            + aten_node_of_interest["imatmul"]
            + aten_node_of_interest["aten.mm"]
            + aten_node_of_interest["conv"]
        ):
            # useful info in node.meta['from_node'], ['nn_module_stack'], and ['source_fn_stack']
            # meta['from_node'] is a list of tuple(s) looks like (fx node name, class/OpPacket)
            # meta['nn_module_stack'] is a dict, {FX_name1: (original_name1, class), },

            if Version(torch.__version__) < Version("2.1"):
                # for PT2.0, fx node name = fx module name(?)
                n_fx_name = list(n_aten.meta["nn_module_stack"].keys())[-1]
            else:
                n_fx_name, _n_fx_class = n_aten.meta["from_node"][0]
            org_mod_name = LUTaten_name_to_org_mod[n_aten.name]
            # NOTE node.meta "original mod name" looks like 'L['args'][0].bert.xxx.yyy[0].zzz
            # L['args'][0] seems to refer to the first arg to torch.compile, which is model itself
            # ModuleList, e.g. yyy[0].zzz, in the LUT should become yyy.0.zzz

            n_aten_AIUname = add_aiu_suffix(n_aten.name)
            # to satisfy AIU naming requirements for addmm/matmul

            preccfg_json["precision"][f"{n_aten_AIUname}/precision"] = "int8"
            preccfg_json["alpha_ap"][f"{n_aten_AIUname}/alpha"] = LUTorg_mod_clipvals[
                f"{org_mod_name}.quantize_feature.clip_val"
            ]
            preccfg_json["alpha_an"][f"{n_aten_AIUname}/alpha"] = LUTorg_mod_clipvals[
                f"{org_mod_name}.quantize_feature.clip_valn"
            ]
            preccfg_json["alpha_wp"][f"{n_aten_AIUname}/kernel"] = LUTorg_mod_clipvals[
                f"{org_mod_name}.quantize_weight.clip_val"
            ]
            preccfg_json["W_int"][f"{n_aten_AIUname}/kernel"] = LUTorg_mod_clipvals[
                f"{org_mod_name}.weight"
            ]
            preccfg_json["input_zp"][f"{n_aten_AIUname}/kernel"] = LUTorg_mod_clipvals[
                f"{org_mod_name}.input_zp"
            ]
            # will handle alpha_wn later
            n_fx = LUTfx_name_to_node[n_fx_name]
            if n_fx.op == "call_module":
                # if the corresponding fx node is a 'call_module', e.g. nn.Linear,
                # can get additional attr from that module if needed, e.g. qmod.in_features
                qmod = gm_fx.get_submodule(n_fx.target)  # pylint: disable=unused-variable

            elif n_fx.op == "call_function":
                # if it's a 'call_func', e.g. torch.ops.fms_mo.imatmul,
                # that means the QLinear module containing this op is decomposed on the FX graph
                # won't be able to "get_submodule", use LUT find cvs
                pass
            else:
                logger.info(
                    f"Cannot find the corresponding FX node of Aten node {n_fx.name}"
                )

        # --- case 2: record for Conv2d/ConvTrans2d
        # --- case 3: record for BMM
        for n_aten in aten_node_of_interest["bmm"]:
            n_fx_name, _n_fx_class = n_aten.meta["from_node"][0]
            n_fx_org_mod_name = list(n_aten.meta["nn_module_stack"].values())[-1][0]
            org_mod_name = lname_to_org_name(n_fx_org_mod_name)
            # bmm name for AIU is just node.name
            n_aten_AIUname = n_aten.name
            n_fx = LUTfx_name_to_node[n_fx_name]
            if n_fx.op == "call_function":
                # QBmm line number can be found in meta['stack_trace'], which has the format of
                # "File xxx, line yyy, in [real code zzz]"
                # e.g., File "transformers/models/bert/modeling_bert.py", line 325, in forward\n
                # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))\n'
                bmm_func_call_trace = n_aten.meta["stack_trace"].split("File ")[-1]
                qbmm_line_no = int(
                    bmm_func_call_trace.split(", ")[1].replace("line ", "")
                )
                org_mod_name += f".QBmm{qbmm_line_no}"

                preccfg_json["precision"][f"{n_aten_AIUname}/precision"] = "int8"
                preccfg_json["alpha_ap"][f"{n_aten_AIUname}/alpha"] = (
                    LUTorg_mod_clipvals[f"{org_mod_name}.quantize_m1.clip_val"]
                )
                preccfg_json["alpha_an"][f"{n_aten_AIUname}/alpha"] = (
                    LUTorg_mod_clipvals[f"{org_mod_name}.quantize_m1.clip_valn"]
                )
                preccfg_json["alpha_wp"][f"{n_aten_AIUname}/kernel"] = (
                    LUTorg_mod_clipvals[f"{org_mod_name}.quantize_m2.clip_val"]
                )
                preccfg_json["alpha_wn"][f"{n_aten_AIUname}/kernel"] = (
                    LUTorg_mod_clipvals[f"{org_mod_name}.quantize_m2.clip_valn"]
                )
                # will handle alpha_wn later

        return gm_aten.forward

    with torch.no_grad():  # no_grad to disable bwd tracing
        gm_fwd = aot_module_simplified(
            gm_fx,  # either implement faketensor tracing or use a gm without torch.quant_per_t
            sample_inp,
            fw_compiler=make_boxed_compiler(fake_fwd_comp),
            decompositions=get_decompositions([torch.ops.aten.rsub.Scalar]),
        )

        # ------- Additional processing of clipval data and update preccfg here -------
        # NOTE: cannot do anything other than recording in aten backend, or it requires FakeTensors
        #       additional processing needed are:
        #       1. turn tensors into python numbers (by Tensor.item())
        #       2. AIU doesn't support per-channel quantization yet, use max() for now
        #       3. calc "zero_shift" from W_int
        #       4. force QKV sync, pick the largest clipval/clipvaln among the 3
        #       5. other adjustment/compensations
        preccfg_json["alpha_ap"] = {
            k: v.item() for k, v in preccfg_json["alpha_ap"].items()
        }
        preccfg_json["alpha_an"] = {
            k: v.item() for k, v in preccfg_json["alpha_an"].items()
        }
        preccfg_json["alpha_wp"] = {
            k: v.max().item() for k, v in preccfg_json["alpha_wp"].items()
        }
        # alpha_wn may have something from bmm already, save the existing ones first
        existing_bmm_alpha_wn = {
            k: v.item() for k, v in preccfg_json["alpha_wn"].items()
        }
        preccfg_json["alpha_wn"] = {
            k: -v for k, v in preccfg_json["alpha_wp"].items()
        }  # this is for sym W
        preccfg_json["alpha_wn"].update(
            existing_bmm_alpha_wn
        )  # overwrite those from bmm

        for k, v in preccfg_json["W_int"].items():
            k_alpha = k.replace("/kernel", "/alpha")
            k_aten = k.replace("/kernel", "").replace("_MatMul", "")
            org_mod_name = LUTaten_name_to_org_mod[k_aten]
            if enforce_clipval_ratio and (
                "attention" in org_mod_name or "output.dense" not in org_mod_name
            ):  # enforce "symmetric activation"
                # TODO review this part again, do not use hard-coded names!
                preccfg_json["alpha_an"][k_alpha] = (
                    preccfg_json["alpha_ap"][k_alpha] * enforce_clipval_ratio
                )
            elif org_mod_name in sym_act:
                logger.info(f"Skipping zero_shift for {org_mod_name}")
            #     # no need to calculate zero_shift for sym act/sym W nodes
            else:
                # without enforcing symmetric activation, we will need a "zero_shift compensation"
                sumdim = tuple(
                    range(1, len(v.shape))
                )  # NOTE (1,) for linear, (1,2,3) for conv
                preccfg_json["zero_shift"][k] = str(
                    v.to(torch.int).clamp(-127, 127).sum(dim=sumdim).tolist()
                )
                # here our INT Ws are saved as FP, cast back to INT for better consistency
        del preccfg_json["W_int"]
        del preccfg_json["input_zp"]

        for grp in QKVsiblings:
            max_cv_in_grp = max(
                preccfg_json["alpha_ap"][f"{grp_i}/alpha"] for grp_i in grp
            )
            min_cvn_in_grp = min(
                preccfg_json["alpha_an"][f"{grp_i}/alpha"] for grp_i in grp
            )
            for grp_i in grp:
                preccfg_json["alpha_ap"][f"{grp_i}/alpha"] = max_cv_in_grp
                preccfg_json["alpha_an"][f"{grp_i}/alpha"] = min_cvn_in_grp

        # additional compensations
        if comp_lay0_by_dropout:
            # user needs to pass dropout prob thru this flag if need compensation
            for k, v in preccfg_json["alpha_ap"].items():
                if "layer_0" in k and any(
                    qkv in k for qkv in ["query", "key", "value"]
                ):
                    preccfg_json["alpha_ap"][k] = v * (1.0 - comp_lay0_by_dropout)

    # Remove any non-serializable key,val pairs from config
    # Local
    from fms_mo.utils.qconfig_utils import serialize_config

    serialize_config(preccfg_json)

    # Finally, output to .json, output_dir is already a Path obj
    preccfg_fname = output_dir / "precconfig.json"

    with open(preccfg_fname, mode="w", encoding="utf-8") as f:
        json.dump(preccfg_json, f, indent=4)
    logger.info(f"AIU meta file {preccfg_fname} is saved!")

    return gm_fwd


def generate_preccfg(
    model,
    sample_inp,
    qcfg,
    tokenizer=None,
    output_dir="preccfg_conversion",
    store_dequantW=False,  # store intW by default
    recalc_w_scale=False,  # sometimes Qmax's clipval is too large, could remove outliers if needed
    _bypass_w_dist_check=False,  # in case some model has non-Gaussian INT W, even SAWB can't fix it
    enforce_clipval_ratio=False,
    comp_lay0_by_dropout=False,
):
    """New, simplified approach to create preccfg for AIU
    ** assuming input model is a fms_mo Qmodel
    1. extract clipvals while reverting QLinear to nn.Linear
        (convert to nn.Linear for now, maybe QLinear_INT_deploy in the future)
    2. trace model and map nodes using node.meta[], collect info for precconfig.json
    3. only support W4A4, W8A8, W32A32 for now, W4A8 W8A4 and etc are not supported
    NOTE:
    a. in QLinear_INT_deploy, we already attached Qa_clip_val, Qa_clip_valn, and Qw_clip_val. Due to
        QLinear can cause tracing issue, must swap to nn.Linear or deployable QLinear wrapper first
        (henec the Qx_clipvals) will be easier to stack/integrate with fms_mo in the near future
    """
    # Standard
    from functools import partial
    from pathlib import Path

    # Third Party
    from torch.ao.quantization.utils import _parent_name
    from torch.nn import Conv2d, ConvTranspose2d, Linear

    # Local
    from fms_mo.modules import QBmm, QConv2d, QConvTranspose2d, QLinear
    from fms_mo.quant.quantizers import PACT, SAWB, PACTplusSym

    tar_dir = Path(output_dir)
    if tar_dir.exists() and tar_dir.is_dir():
        raise RuntimeError(
            f"Directory {output_dir} already exists, please double check to avoid overwritting !!"
        )
    tar_dir.mkdir(parents=True, exist_ok=True)

    # Ensure model contains at least one quantized module
    # Local
    from fms_mo.prep import has_quantized_module

    if not has_quantized_module(model):
        raise RuntimeError(f"Model {model} does not have any quantized modules")

    # NOTE make sure the model is not under training, otherwise some of the quantizers
    # will keep updating the clipvals!
    model.eval()

    mod_map = {QLinear: Linear, QConv2d: Conv2d, QConvTranspose2d: ConvTranspose2d}
    LUTcvs_and_iW = {n: b for n, b in model.state_dict().items() if "clip_val" in n}
    LUTqmod = {"linear": [], "conv": [], "bmm": []}
    # --- [step 1] extract clipvals while reverting QLinear to nn.Linear
    #       can choose to store INT weights or dequant weights,
    #       If using dequant weights, FP eval results will look closer to "normal",
    #       but the precconfig.json we saved are still based on INT weights/scales...
    target_device = next(model.parameters()).device
    sym_act = []  # these nodes will skip zero_shift later
    mod_at_risk_aiu = []  # layers could cause underflow on AIU due to intW being too small
    Nmod_reset = 0
    for n, m in model.named_modules():
        m.to(target_device)  # make sure module is on the same device
        if isinstance(m, tuple(mod_map.keys())):
            assert all(
                getattr(m, a_or_w) in [4, 8, 32]
                for a_or_w in ["num_bits_feature", "num_bits_weight"]
            ), "Please check nbits setting!"
            parent_name, module_name = _parent_name(n)
            parent_mod = model.get_submodule(parent_name)
            fms_mo_w_dtype = m.weight.dtype

            base_params = {}
            if hasattr(m, "__constants__"):
                base_params = {k: getattr(m, k) for k in m.__constants__}
                base_params["bias"] = m.bias is not None
            else:
                raise RuntimeError(
                    f"Module {n}, class {type(m)}, doesn't have required constants. Please check."
                )
            base_params["device"] = next(m.parameters()).device
            base_params["dtype"] = fms_mo_w_dtype
            if isinstance(m, Conv2d):  # ConvTrans?
                if "output_padding" in base_params:
                    base_params.pop(
                        "output_padding"
                    )  # nn.Conv2d doesn't like this param ...
                if base_params["padding_mode"] != "zeros":
                    logger.info(
                        f"Warning! {n} padding mode = {base_params['padding_mode_i']}"
                    )
                # isDW = m.in_channels == m.out_channels and m.groups == m.in_channels

            nnmod = mod_map[type(m)](**base_params)
            if m.num_bits_weight == 32 and m.num_bits_feature == 32:
                nnmod.weight = m.weight
                if m.bias is not None:
                    nnmod.bias = m.bias
                setattr(parent_mod, module_name, nnmod)
                continue

            # prepare for INT W and scales/zp
            with torch.no_grad():
                Qa = m.quantize_feature
                if isinstance(Qa, PACT):
                    cvn = torch.zeros_like(Qa.clip_val)
                    LUTcvs_and_iW[f"{n}.quantize_feature.clip_valn"] = (
                        cvn  # add a fake entry
                    )
                elif isinstance(Qa, PACTplusSym) or (
                    hasattr(Qa, "minmax") and Qa.minmax is False
                ):
                    # PACTplusSym typically assumes +- clipval with 2**n-2 levels
                    # if Qmax is used for act, only when self.minmax = False it becomes symmetric
                    cvn = (
                        -Qa.clip_val / 127 * 128
                    )  # if enforce_clipval_ratio else -Qa.clip_val
                    LUTcvs_and_iW[f"{n}.quantize_feature.clip_valn"] = (
                        cvn  # add a fake entry
                    )
                    sym_act.append(n)
                else:
                    cvn = Qa.clip_valn
                input_scale = (Qa.clip_val - cvn) / (2**Qa.num_bits - 1)
                input_zero_point = torch.round(-cvn / input_scale).to(torch.int)
                LUTcvs_and_iW[f"{n}.input_scale"] = input_scale
                LUTcvs_and_iW[f"{n}.input_zp"] = input_zero_point

                # --- use SAWB (recalculate clipvals) or use the quantizer in model
                if recalc_w_scale and m.num_bits_weight == 8:
                    # if Qmax pickup wild outlier, it may affect AIU accuracy
                    Qw = SAWB(m.num_bits_weight, align_zero=True).to(target_device)
                    Qw.train()
                else:
                    Qw = m.quantize_weight
                    Qw.eval()
                # NOTE if SAWB is under training mode, calling .forward() will trigger recalc of
                # clipvals.  The original stored clipvals (from training/tuning) would be ignored
                Qw.dequantize = False
                w_int8 = Qw(
                    m.weight.float()
                )  # if Qw is SAWB, calling Qw.fwd() will update clipval one more time (intended)
                w_scale = Qw.clip_val * 2 / (2**Qw.num_bits - 2)

                # make sure intW is not "narrowly distributed" in INT space,
                # TODO "20" is empirical, will need to adjust later
                if w_int8.var().sqrt() < 20 and not _bypass_w_dist_check:
                    mod_at_risk_aiu.append(n)
                    message = (
                        f"{n},{m} is narrowly distributed with weight std dev ="
                        f"{w_int8.var().sqrt()}"
                    )
                    warnings.warn(message, UserWarning)

                w_zp = torch.zeros_like(w_scale, dtype=torch.int)
                LUTcvs_and_iW[f"{n}.quantize_weight.clip_val"] = (
                    Qw.clip_val
                )  # update W clipval in LUT just in case
                LUTcvs_and_iW[f"{n}.w_scale"] = w_scale.to(fms_mo_w_dtype)
                LUTcvs_and_iW[f"{n}.w_zp"] = w_zp
                # corr_term = (
                #     input_scale.float()
                #     * (input_zero_point - 128)
                #     * w_int8.sum(dim=1)
                #     * w_scale.float()
                # )
                # dim=1 because w_int is in [out,in], after sum shape=[out,]
                # same as w_scale and bias.
                # (zp-128)*w_int8.sum(dim=1) can be >> fp16.max, use fp32 scales to make sure
                # dtype is large enough
                if m.bias is not None:
                    nnmod.bias = m.bias

                # NOTE: for a deployable linear, bias should -= corr_terms, In current AIU settings,
                #       we provide the correction term in precconfig.json instead
                if store_dequantW:
                    if isinstance(m, QLinear):
                        nnmod.weight = torch.nn.Parameter((w_int8.t() * w_scale).t())
                    elif (
                        isinstance(m, (QConv2d, QConvTranspose2d))
                        and len(w_scale.shape) == 1
                    ):
                        # only support perT for now
                        nnmod.weight = torch.nn.Parameter(w_int8 * w_scale)
                else:
                    nnmod.weight = torch.nn.Parameter(
                        w_int8.float(), requires_grad=False
                    )  # NOTE: current AIU settings prefer to take INT W stored as FP...

            # update LUT directly if needed
            LUTcvs_and_iW[f"{n}.weight"] = w_int8.detach()

            setattr(parent_mod, module_name, nnmod)

            if isinstance(m, QLinear):
                LUTqmod["linear"].append(n)
            elif isinstance(m, (QConv2d, QConvTranspose2d)):
                LUTqmod["conv"].append(n)
            Nmod_reset += 1

        elif isinstance(m, QBmm):
            assert (
                m.num_bits_m1 in [7, 8] and m.num_bits_m2 == 8
            ), "Only support 8bit QBmm!"
            LUTqmod["bmm"].append(n)
            with torch.no_grad():
                Qa = m.quantize_m1
                Qw = m.quantize_m2
                cvn_ext_ratio = (
                    128 / 127
                )  # may trained as sym (-127,127) but can use extended range (-128,127)
                if m.num_bits_m1 == 7 or "sym" in m.qm1_mode:
                    Qa_cvn = -Qa.clip_val
                else:
                    Qa_cvn = getattr(
                        Qa, "clip_valn", torch.zeros_like(Qa.clip_val)
                    )  # even if softmax, still want to avoid using 0...
                m1_scale = (Qa.clip_val - Qa_cvn) / (2**Qa.num_bits - 1)
                m2_scale = Qw.clip_val * 2 / (2**Qw.num_bits - 2)
                # update LUT directly
                LUTcvs_and_iW[f"{n}.quantize_m1.scale"] = m1_scale
                LUTcvs_and_iW[f"{n}.quantize_m1.zp"] = torch.round(
                    -Qa_cvn / m1_scale
                ).to(torch.int)
                LUTcvs_and_iW[f"{n}.quantize_m2.scale"] = m2_scale
                LUTcvs_and_iW[f"{n}.quantize_m2.zp"] = torch.zeros_like(
                    m2_scale, dtype=torch.int
                )

                LUTcvs_and_iW[f"{n}.quantize_m1.clip_val"] = Qa.clip_val.detach()
                LUTcvs_and_iW[f"{n}.quantize_m1.clip_valn"] = (
                    Qa_cvn.detach() * cvn_ext_ratio
                )
                LUTcvs_and_iW[f"{n}.quantize_m2.clip_val"] = Qw.clip_val.detach()
                LUTcvs_and_iW[f"{n}.quantize_m2.clip_valn"] = (
                    -Qw.clip_val.detach() * cvn_ext_ratio
                )  # always sym
        m.to("cpu")

    logger.info(
        f"Reset {Nmod_reset} modules from QLinear to nn.Linear or QConv to nn.Conv."
    )
    qbmm2delete = [n for n, m in model.named_modules() if isinstance(m, QBmm)]
    for n in qbmm2delete:
        parent_name, module_name = _parent_name(n)
        parent_mod = model.get_submodule(parent_name)
        delattr(parent_mod, module_name)
    logger.info(f"Delete {len(qbmm2delete)} QBmm modules.")

    # prep a look-up table for fx level name->original names, only needed for precconfig generation
    model.to(target_device).eval()
    LUTmod2org_name = {
        m: n for n, m in model.named_modules() if isinstance(m, tuple(mod_map.values()))
    }
    cus_bknd = partial(
        cus_backend_generate_preccfg,
        LUTorg_mod_clipvals=LUTcvs_and_iW,
        LUTquant_mods=LUTqmod,
        LUTmod2org_name=LUTmod2org_name,
        enforce_clipval_ratio=enforce_clipval_ratio,
        comp_lay0_by_dropout=comp_lay0_by_dropout,
        output_dir=tar_dir,
        mappables=tuple(mod_map.values()),
        sym_act=sym_act,
    )

    # --- [step 2] trace the model and collect node info, ONLY NEEDED for INT8/INT4 cases
    #               then generate preccfg and save the full model
    if qcfg["nbits_w"] != 32 and qcfg["nbits_a"] != 32:
        model_compiled = torch.compile(model.forward, backend=cus_bknd, fullgraph=True)
        with torch.no_grad():
            if isinstance(sample_inp, (dict, BatchEncoding)):
                sample_inp = {k: v.to(target_device) for k, v in sample_inp.items()}
                model_compiled(**sample_inp)
            elif isinstance(sample_inp, tuple):
                model_compiled(*sample_inp)
            elif isinstance(sample_inp, torch.Tensor):
                model_compiled(sample_inp)
            else:
                logger.info(
                    "Sample input provided by user is not in [dict, tuple, tensor], "
                    "please double check!"
                )
        # we don't really need the compiled model, json is saved
        del model_compiled
        torch._dynamo.reset()

    # --- [Final step] save the full uncompiled model and/or state_dict() if needed
    # Current AIU settings does not expect QBmm module to be called nor appear on the graph.
    # Simply remove them from the model here and make sure we don't use context manager

    # NOTE Guard rail for AIU ---
    if len(mod_at_risk_aiu) > 0:
        logger.info(
            "Some layers have narrowly distributed INT W, may be caused by "
            "non-ideal W scaling factor. It will likely result in poor accuracy on AIU. \n"
            "Please consider using this flag\n"
            "    generate_preccfg(..., rescalc_w_scale=True)"
            "If the flag is already True but this error still shows up, please contact quantization"
            "team for further investigation.\n"
            "Model is not converted successfully and no checkpoint is saved!"
        )
        raise RuntimeError(
            "Some modules are at risk for AIU due to INT W distribution. No checkpoint is saved."
        )

    if hasattr(model, "config") and hasattr(model, "save_pretrained"):
        # for Huggingface models, use built-in save_pretrained()
        model.save_pretrained(output_dir)
        if tokenizer:
            tokenizer.save_pretrained(output_dir)
        logger.info("Hugginface model is saved successfully.")

    # NOTE currently we also save a full model for comparison on AIU , could remove in the future
    torch.save(model, tar_dir / "model_new_aiu_intW.pt")
    logger.info(
        f"Full model 'model_new_aiu_intW.pt' is saved successfully under {output_dir}."
    )

    # Local
    from fms_mo.utils.qconfig_utils import qconfig_save

    qconfig_save(qcfg, tar_dir / "qcfg.json")
    logger.info(f"{tar_dir/'qcfg.json'} is also saved.")


def cus_backend_verify_preccfg(
    gm_fx,
    sample_inp,
    LUTmod2org_name=None,
):
    """main purpose is to find mapping between "aten node" <-> "original module" """
    # Third Party
    from torch._decomp import get_decompositions
    from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_compiler

    # LUTfx_name_to_node = {n.target: n for n in gm_fx.graph.nodes}
    global LUTaten_name_to_org_mod  # pylint: disable=global-statement
    LUTaten_name_to_org_mod = {}
    mappable = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d)
    if Version(torch.__version__) < Version("2.1"):
        LUTmod_name_fx2org = {
            n_fx: LUTmod2org_name[m]
            for n_fx, m in gm_fx.named_modules()
            if isinstance(m, mappable)
        }
        LUTmod_name_fx2org2 = {}
    else:
        # Newer PT alters the Linear instance, a given mod in FX gm may still look like Linear, but
        # id(mod) != id(original_mod_before_dynamo)
        # We could use id(mod.weight) to match, or make a guess from Lname
        # option 1:
        LUTmod_name_fx2org = {
            n_fx: lname_to_org_name(n_fx)
            for n_fx, m in gm_fx.named_modules()
            if isinstance(m, mappable)
        }
        # option 2: something like "getattr_L__self___layer4___2___conv3" could fail in Option 1...
        # in some cases it's not fully flattened, like
        # "L__self___encoder_layers_encoder_layer_0_self_attention.out_proj"
        # NOTE matching module itself (i.e. pointer addr, id()) could be inconsistent sometime,
        #      but address of "mod.weight" should be consistent.
        LUTmodW2org_name = {m.weight: n for m, n in LUTmod2org_name.items()}
        LUTmod_name_fx2org2 = {
            n_fx: LUTmodW2org_name[m.weight]
            for n_fx, m in gm_fx.named_modules()
            if isinstance(m, mappable)
        }
        # TODO consider to use option 2 to replace option 1 in the future, OR use the approach
        # in generate_preccfg(), i.e. utilize node.meta

    mod_cant_be_mapped = []
    update_dict = {}
    for k, v in LUTmod_name_fx2org.items():
        if v is None:
            dblchk = LUTmod_name_fx2org2.get(k, None)
            if dblchk is None:  # failed both option 1 and 2 -> cannot be mapped
                mod_cant_be_mapped.append(k)
            else:  # update LUT
                update_dict[k] = dblchk
    if len(update_dict) > 0:
        LUTmod_name_fx2org.update(update_dict)

    assert (
        not mod_cant_be_mapped
    ), f"some fx modules cannot be mapped to its origin: {mod_cant_be_mapped}"

    def fake_fwd_comp(
        gm_aten,
        inp,  # pylint: disable=unused-argument
    ):
        # First, find target nodes in Aten GM and do some clean-up, e.g. QKV sync and QBmm check
        # NOTE qbmm is not in the gm_fx, mapping of bmm Op to QBmm is done in the next part
        # fms_mo.imatmul is a custom Op we register for QLinearINT8Deploy
        aten_node_of_interest = {
            "addmm": [],
            "conv": [],
            "bmm": [],
            "imatmul": [],
            "aten.mm": [],
        }
        for n in gm_aten.graph.nodes:
            for op_kw, _node_list in aten_node_of_interest.items():
                if op_kw in str(n.target):
                    op_type = (
                        "linear" if op_kw in ["addmm", "imatmul", "aten.mm"] else op_kw
                    )
                    if Version(torch.__version__) < Version("2.1"):
                        # for PT2.0
                        fx_mod_name = list(n.meta["nn_module_stack"].keys())[-1]
                        org_mod_name = LUTmod_name_fx2org.get(fx_mod_name, None)
                    else:
                        # for newer PT
                        n_fx_org_mod_name = list(n.meta["nn_module_stack"].values())[
                            -1
                        ][0]
                        org_mod_name = lname_to_org_name(n_fx_org_mod_name)

                    if op_type == "linear":
                        # to satisfy AIU naming requirements for addmm/matmul
                        n_aten_AIUname = add_aiu_suffix(n.name)
                    else:
                        n_aten_AIUname = n.name
                    LUTaten_name_to_org_mod[n_aten_AIUname] = (
                        org_mod_name  # add aiu name to the LUT as well
                    )

        return gm_aten.forward

    # no_grad to disable bwd tracing
    with torch.no_grad():
        gm_fwd = aot_module_simplified(
            gm_fx,  # either implement faketensor tracing or use a gm without torch.quant_per_t
            sample_inp,
            fw_compiler=make_boxed_compiler(fake_fwd_comp),
            decompositions=get_decompositions([torch.ops.aten.rsub.Scalar]),
        )

    logger.info("Mapping for aten_node_name to original_module_name is done!")

    return gm_fwd  # no real modifications were made to the model, but need to rerutn a 'callable'


class ActCompHook(torch.nn.Module):
    """
    Store hooks for activation

    Extends:
        torch.nn.Module
    """

    def __init__(self, mod_name, summary_dict, whichmodel, cache_dev="cuda"):
        super().__init__()
        self.mod_name = mod_name
        self.cache_dev = cache_dev
        self.sum_dict = summary_dict  # keep a reference instead of using global
        self.whichmodel = whichmodel  # either 'ref' or 'new'

    def __call__(self, mod_calling, inp, out):
        # usually input is a tuple, output is a tensor
        # if compare only one batch of data, keep it on GPU
        with torch.no_grad():
            if hasattr(mod_calling, "quantize_feature"):
                Qa = mod_calling.quantize_feature
                # Qw = mod_calling.quantize_weight
                cva = Qa.clip_val
                cvna = Qa.clip_valn
                # cvw = Qw.clip_val
                nbits = mod_calling.num_bits_weight
                scale_a = (cva - cvna) / (2**nbits - 1)
                zp_a = torch.round(-cvna / scale_a).to(torch.int)
                # scale_w = cvw / (2 ** (nbits - 1) - 1)
                in_dq = Qa(inp[0])
                Qa.dequantize = False
                in_q = Qa(inp[0]).to(inp[0].dtype)
                Qa.dequantize = True
            else:
                # make sure qlin_veri and _int8 all have .cvs and .nbits_w
                nbits = getattr(mod_calling, "nbits_w")
                scale_a = mod_calling.input_scale
                zp_a = mod_calling.input_zp
                # scale_w = mod_calling.w_scale
                cva, cvna, _cvw = torch.tensor(
                    mod_calling.cvs,  # stored as a list of python float
                    device=zp_a.device,
                ).split(1)
                if mod_calling.usePTnativeQfunc:
                    in_q = torch.clamp((inp[0] / scale_a + zp_a).round(), 0, 255)
                    in_dq = scale_a * (in_q - zp_a)
                else:
                    # fms_mo has slightly different def for scale and zp
                    quant_scale = getattr(
                        mod_calling, "quant_scale", (2**nbits - 1) / (cva - cvna)
                    )
                    quant_stepsize = getattr(
                        mod_calling, "quant_stepsize", 1.0 / quant_scale
                    )
                    quant_zero_point = getattr(
                        mod_calling, "quant_zero_point", torch.round(cvna * quant_scale)
                    )
                    in_q = torch.round(
                        inp[0].clamp(cvna, cva) / quant_stepsize - quant_zero_point
                    )
                    in_dq = (in_q + quant_zero_point) * quant_stepsize
                # in Qlinear_aiu_veri/_int8_deploy forward, will check this flag and
                # choose the right math

            # NOTE final out = mm output (dq if INT mm) + (bias - corr_term)

            if self.mod_name not in self.sum_dict:
                # this layer is called for the first time, save the activations
                self.sum_dict[self.mod_name] = {
                    "in": inp[0].detach().to(self.cache_dev),
                    "out": out.detach().to(self.cache_dev),
                    "in_q": in_q.detach().to(self.cache_dev),
                    "in_dq": in_dq.detach().to(self.cache_dev),
                }
            else:
                # called by the second model (usually ref), calc delta/norm
                currDev = out.device
                self.sum_dict[self.mod_name]["in"] = torch.norm(
                    self.sum_dict[self.mod_name]["in"].to(currDev) - inp[0]
                ) / torch.norm(inp[0])
                self.sum_dict[self.mod_name]["out"] = torch.norm(
                    self.sum_dict[self.mod_name]["out"].to(currDev) - out
                ) / torch.norm(out)
                self.sum_dict[self.mod_name]["in_q"] = torch.norm(
                    self.sum_dict[self.mod_name]["in_q"].to(currDev) - in_q
                ) / torch.norm(in_q)
                self.sum_dict[self.mod_name]["in_dq"] = torch.norm(
                    self.sum_dict[self.mod_name]["in_dq"].to(currDev) - in_dq
                ) / torch.norm(in_dq)


def compare_models(
    model,
    model_ref,
    sample_inp,
    target_device="cuda",
    output_filename="model_comparison_summary",
):
    """for verification purpose, compare the 2 models user provided,
    "reference model" or "model_ref" has to be a qmodel,
    "the other" or "model" can be q/dq/mm lvl-0, or q/mm/dq lvl-1/2
    will check:
    1. difference in weights and clipvals
    2. attach hooks to both models to compare the input/output activations
    3. save the results to a pt for visualization
    NOTE only works on QLinear (transformers) right now
    """
    # Third Party
    import pandas as pd

    # Local
    from fms_mo.modules import QLinear

    model_ref.eval()
    model.eval()
    # ref_sd = model_ref.state_dict()
    # new_sd = model.state_dict()
    summary_dict = {
        "layer": [],
        "||diff_qW||/||qW||": [],
        "||diff_dqW||/||dqW||": [],
        "||diff corr_term||": [],
        "||diff bias||": [],
        "diff scale_w": [],
        "diff scale_a": [],
        "diff zp_a": [],
    }
    summary_act = {}
    hook_handles = []
    nbits = 8  # only support 8bits for now

    # make sure reference is a Qmodel
    qlins_ref = {n: m for n, m in model_ref.named_modules() if isinstance(m, QLinear)}
    assert (
        len(qlins_ref) > 0
    ), "model_ref in compare_models() does not have any QLinear to compare with."

    # --- 1. compare weights and clipvals
    with torch.no_grad():
        for mod_name, mod_ref in qlins_ref.items():
            mod_new = model.get_submodule(mod_name).to(target_device)
            # make sure the two module are on the same device, so we can get consistent results
            mod_ref = mod_ref.to(target_device)

            Qw_ref = mod_ref.quantize_weight
            Qa_ref = mod_ref.quantize_feature
            scale_w_ref = 2 * Qw_ref.clip_val / (2**nbits - 2)
            if getattr(mod_new, "usePTnativeQfunc", True):
                scale_a_ref = (Qa_ref.clip_val - Qa_ref.clip_valn) / (2**nbits - 1)
                zp_a_ref = torch.round(-Qa_ref.clip_valn / scale_a_ref)
            else:
                quant_scale = (2**nbits - 1) / (Qa_ref.clip_val - Qa_ref.clip_valn)
                quant_stepsize = 1.0 / quant_scale
                quant_zero_point = torch.round(Qa_ref.clip_valn * quant_scale)
                scale_a_ref = quant_stepsize
                zp_a_ref = -quant_zero_point

            # NOTE if new model is a lvl1/2 model => won't use QLinear, i.e. no quantizers nor
            # clipvals, W is qW. Otherwise, new model is lvl0, W is dq'ed from INT W
            newMod_has_dqW = isinstance(mod_new, QLinear)

            # --- 1-1. compare scales and zp, ie, not clipvals
            scale_w_new = None
            scale_a_new = None
            zp_a_new = None
            cvw_new = None
            if newMod_has_dqW:
                # case i: comparing level0 vs original fms_mo model
                Qw_new = mod_new.quantize_weight
                Qa_new = mod_new.quantize_feature
                cva_new = Qa_new.clip_val
                cvna_new = Qa_new.clip_valn
                # this may introduce some rounding error
                scale_w_new = 2 * Qw_new.clip_val / (2**nbits - 2)
                scale_a_new = (cva_new - cvna_new) / (2**nbits - 1)
                zp_a_new = torch.round(-cvna_new / scale_a_new)
            elif hasattr(mod_new, "input_scale"):
                # case ii: this is level 1/2 compared with level0
                scale_w_new = mod_new.w_scale
                scale_a_new = mod_new.input_scale
                zp_a_new = mod_new.input_zp
                if hasattr(mod_new, "cvs"):
                    # cva and cvw from preccfg are stored already
                    cvw_new = torch.tensor(mod_new.cvs[2], device=scale_w_new.device)
                else:
                    cvw_new = scale_w_new * (2**nbits - 2) / 2

            summary_dict["layer"].append(mod_name)
            summary_dict["diff scale_w"].append((scale_w_new - scale_w_ref).item())
            summary_dict["diff scale_a"].append((scale_a_new - scale_a_ref).item())
            summary_dict["diff zp_a"].append((zp_a_new - zp_a_ref).item())

            # --- 1-2. compare qW and dqW
            # NOTE only as-trained/tuned model_ref has 'continuous W',
            #      level 0 has dqW from INT W loaded from file, level 1/2 stores qW directly
            if newMod_has_dqW:
                new_dqW = mod_new.weight
                Qw_new.dequantize = False
                new_qW = Qw_new(new_dqW)
                Qw_new.dequantize = True
            else:
                new_qW = mod_new.weight

            ref_dqW = Qw_ref(mod_ref.weight)
            # ref_dqW = mod_ref.weight
            # NOTE old fms_mo quantizer has different formula, i.e. clamp before rounding
            if getattr(mod_new, "usePTnativeQfunc", True):
                # "correct" formula, i.e. round then clamp
                ref_qW = (
                    torch.round(ref_dqW / scale_w_ref).clamp(-127, 127).to(torch.int8)
                )
                new_dqW = new_qW * scale_w_new
            else:
                # try to use fms_mo formula
                Qw_ref.dequantize = False
                ref_qW = Qw_ref(mod_ref.weight)  # .to(torch.int8)
                Qw_ref.dequantize = True
                if not newMod_has_dqW:
                    new_qW0254 = new_qW.float() + (2 ** (nbits - 1) - 1)
                    new_dqW = (2 * (new_qW0254 / (2**nbits - 2)) - 1) * cvw_new
                    # NOTE this formula should reduce to qW*(cvw/ (2**(n-1) -1) ), ie qW * scale_w
                    # Apparently we will get some extra rounding error from this fms_mo formula.

            summary_dict["||diff_qW||/||qW||"].append(
                (
                    torch.norm((new_qW - ref_qW).float()) / torch.norm(ref_qW.float())
                ).item()
            )
            summary_dict["||diff_dqW||/||dqW||"].append(
                (torch.norm(new_dqW - ref_dqW) / torch.norm(ref_dqW)).item()
            )

            # --- 1-3. compare bias and recalc correction term and compare
            corr_term_recalc = (
                (zp_a_new - 128)
                * (mod_new.weight.t().sum(dim=0))
                * scale_w_new
                * scale_a_new
            )
            # NOTE NOTE different sequence will cause different rounding errors,
            #       eg, sw*sa*(zp-128)*zero_shift != sa*(zp-128)*zero_shift*sw ??
            #       in QLin_aiu_veri we use (input_zero_point-128)*zero_s*w_scale*input_scale
            summary_dict["||diff bias||"].append(
                torch.norm(mod_new.bias - mod_ref.bias).item()
            )
            if zp_a_new == 128:
                summary_dict["||diff corr_term||"].append(0)  # avoid div by 0
            else:
                summary_dict["||diff corr_term||"].append(
                    (
                        torch.norm(mod_new.corr_term - corr_term_recalc)
                        / torch.norm(mod_new.corr_term)
                    ).item()
                )

            # --- 1-4. attach hooks for later use
            hook_handles.append(
                mod_ref.register_forward_hook(
                    ActCompHook(mod_name, summary_act, "ref", target_device)
                )
            )
            hook_handles.append(
                mod_new.register_forward_hook(
                    ActCompHook(mod_name, summary_act, "new", target_device)
                )
            )

        # --- 2. compare activations thru hooks, will compare input/output
        # Make sure everything is run on target device for consistency
        # at the same time, be conservative on memory usage
        model_ref.to("cpu")
        if isinstance(sample_inp, (dict, BatchEncoding)):
            model.to(target_device)
            mod_new_out = model(**sample_inp)
            model.to("cpu")

            model_ref.to(target_device)
            mod_ref_out = model_ref(**sample_inp)
            model_ref.to("cpu")
        else:
            raise RuntimeError(
                "Model comparison func only supports transformers. Add implementations if needed."
            )

    for h in hook_handles:
        h.remove()

    # rearrange data collected by activation hook for better printing
    sum_dict_fmt = {
        "layer": [],
        "in_q": [],
        "in_dq": [],
        "||diff in||/||in||": [],
        "||diff out||/||out||": [],
    }
    for k, v in summary_act.items():
        sum_dict_fmt["layer"].append(k)
        sum_dict_fmt["in_q"].append(v["in_q"].item())
        sum_dict_fmt["in_dq"].append(v["in_dq"].item())
        sum_dict_fmt["||diff in||/||in||"].append(v["in"].item())
        sum_dict_fmt["||diff out||/||out||"].append(v["out"].item())
    # add the final output
    for out_item in ["loss", "start_logits", "end_logits", "logits"]:
        if hasattr(mod_new_out, out_item):
            item_new = getattr(mod_new_out, out_item)
            item_ref = getattr(mod_ref_out, out_item)
            if item_new is not None and item_ref is not None:
                sum_dict_fmt["layer"].append("final " + out_item)
                sum_dict_fmt["in_q"].append("nan")
                sum_dict_fmt["in_dq"].append("nan")
                sum_dict_fmt["||diff in||/||in||"].append("nan")
                sum_dict_fmt["||diff out||/||out||"].append(
                    (torch.norm(item_new - item_ref) / torch.norm(item_ref)).item()
                )

    df = pd.DataFrame(summary_dict)
    dfhook = pd.DataFrame(sum_dict_fmt)
    df = df.merge(dfhook, on="layer", how="right")
    df.to_csv(output_filename + ".csv")
    logger.info(f"Model comparison is done and saved to {output_filename}.csv.")
    logger.info(df)

    del model_ref


# define as global so verify_preccfg + custom backend can modify it
LUTaten_name_to_org_mod = {}


def verify_preccfg(
    path_to_model_files,
    sample_inp,
    qcfg=None,
    preccfg_json=None,  # filepath to the json, if it's not under model path or not default name
    sim_level=0,  # level0 = qmodel_prep (fake quant), level1=Qlinear_aiu_veri (fake imatmul)
    model_ref=None,
    compare_with_ref=False,
    **kwargs,
):
    """assume the model the full model we saved by generate_precfg()
    we will load from .pt (i.e. INT W saved in nn.Linear)
    together with preccfg, we can rebuild the Qmodel, as if from qmodel_prep,
    it can be used for further eval/test
    NOTE, do not support QBmm yet
    """
    # Standard
    from copy import deepcopy
    from functools import partial
    from pathlib import Path

    # Third Party
    from torch.ao.quantization.utils import _parent_name

    # Local
    from fms_mo import qconfig_init, qmodel_prep
    from fms_mo.modules.linear import QLinearINT8Deploy

    # NOTE for some reason tracing on CPU is unreasonably SLOW, it happened after custom
    # backend returns gm_fwd as a temp patch, do it on CUDA then move back to CPU right afterward
    target_device = kwargs.get("device", "cuda")
    # --- Step 0: load precconfig.json and full model, i.e. saved by generate_preccfg()
    if issubclass(type(path_to_model_files), torch.nn.Module):
        model = path_to_model_files
        if preccfg_json:
            preccfg_file = Path(preccfg_json)
        else:
            raise RuntimeError("Precconfig file cannot be found. Please check.")

    elif isinstance(path_to_model_files, str):
        path_to_files = Path(path_to_model_files)
        if not path_to_files.is_dir():
            raise RuntimeError(
                "Path_to_model_files is not a valid directory. Please check."
            )

        pt_files = list(path_to_files.glob("*.pt"))
        pt_bin_files = list(path_to_files.glob("*.bin"))
        safetensors_files = list(path_to_files.glob("*.safetensors"))
        preccfg_file = list(path_to_files.glob("precconfig*.json"))[0]
        qcfg_file = list(path_to_files.glob("qcfg*.json"))
        # 0-1. try to load model for verification
        if len(safetensors_files) > 0 or len(pt_bin_files) > 0:
            # attempt to use Hugginface to load the model
            # temporary hack, assuming QA model (ie BERT) only for now
            # Third Party
            from transformers import AutoModelForQuestionAnswering  # AutoConfig,

            # config = AutoConfig.from_pretrained(path_to_files)  # architectures
            model = AutoModelForQuestionAnswering.from_pretrained(path_to_files)
        elif len(pt_files) > 0:
            # a full model saved by torch.save()
            model = torch.load(pt_files[0], map_location=target_device)
        else:
            raise RuntimeError(
                f"Cannot find model files under {path_to_model_files}. Please check."
            )

    else:
        raise RuntimeError(
            f"Model path {path_to_model_files} cannot be resolved. Please check."
        )

    # 0-2. load precconfig.json
    with open(preccfg_file, "r", encoding="utf-8") as openfile:
        preccfg = json.load(openfile)
    # 0-3. if qcfg.json exist, also load qcfg.json, otherwise use default
    if qcfg is None:
        qcfg = qconfig_init()

    if len(qcfg_file) > 0:
        # use loaded qcfg to overwrite user-provided qcfg, if both has real values
        # more often the one provided by user is less accurate?
        with open(qcfg_file[0], "r", encoding="utf-8") as openfile:
            qcfg_loaded = json.load(openfile)
        logger.warning(
            "A qcfg dict is provided thru args but the ckpt folder also has "
            "a qcfg.json. Will use the json's value if both exist!!"
        )
        for k, v in qcfg.items():
            if qcfg_loaded.get(k, None) and qcfg_loaded[k] != v:
                qcfg[k] = qcfg_loaded[k]
                logger.warning(
                    f"qcfg[{k}] = {v} (user provided) and {qcfg_loaded[k]} (loaded from file.)"
                )
        logger.warning(
            f"Missing keys in qcfg file and user provided qcfg "
            f"{set(qcfg.keys()) - set(qcfg_loaded.keys())}"
        )

    assert qcfg is not None and len(qcfg) > 0, "qcfg is None or empty..."

    if qcfg["qmodel_calibration_new"] != 0 or qcfg["qmodel_calibration"] != 0:
        qcfg["qmodel_calibration_new"] = 0
        qcfg["qmodel_calibration"] = 0
        logger.warning(
            "qcfg['qmodel_calibration_new'] was not 0 and has been set to 0 now. "
            "We do not want to run calibration during verification!"
        )

    # --- Step 1: prep a look-up table for module name <-> aten/aiu name, BMM is not supported yet
    LUTmod2org_name = {
        m: n
        for n, m in model.named_modules()
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d))
    }

    cus_bknd = partial(
        cus_backend_verify_preccfg,
        LUTmod2org_name=LUTmod2org_name,
    )
    model.to(target_device).eval()
    model_compiled = torch.compile(model.forward, backend=cus_bknd)
    with torch.no_grad():
        if isinstance(sample_inp, (dict, BatchEncoding)):
            sample_inp = {k: v.to(target_device) for k, v in sample_inp.items()}
            model_compiled(**sample_inp)
        elif isinstance(sample_inp, tuple):
            model_compiled(*sample_inp)
        elif isinstance(sample_inp, torch.Tensor):
            model_compiled(sample_inp)
        else:
            logger.info(
                "Sample input provided by user is not in [dict, tuple, tensor], "
                "please double check!"
            )
    # we don't really need the compiled model, just need LUTaten_name_to_org_mod
    del model_compiled

    # --- Step 2: verification begins.
    #             can choose from two level of verification.
    #               level 0 = use fake quant to simulate aiu behavior, i.e. Q->dQ->matmul
    #                         this level WILL NOT utilize "zero_shift"
    #               level 1 = use fake imatmul to simulate aiu behavior, i.e. Q->matmul->dQ,
    #                         this level WILL utilize "zero_shift"
    # currently precconfig only supports perT for W
    if "perCh" in qcfg["qw_mode"]:
        qcfg["qw_mode"] = qcfg["qw_mode"].replace("perCh", "")
        logger.warning(
            "Quantization config is using perCh for weight, but AIU only supports perT for now. "
            "This setting is changed back to perT now."
        )

    node_used = []
    nodes_in_preccfg = [
        k.replace("/precision", "") for k in preccfg["precision"].keys()
    ]
    LUTorg_mod_to_aiu_name = {
        mod_name: aten_name for aten_name, mod_name in LUTaten_name_to_org_mod.items()
    }

    model_intW_backup = None
    if not model_ref:
        # will use dQ W (level0 outcome) for lvl1 and 2
        # TODO be careful with memory management, may need to move to CPU before deepcopy
        model_intW_backup = deepcopy(model).cpu()

    # recreate the qmodel for all sim levels
    qmodel_prep(model, sample_inp, qcfg, use_dynamo=True)

    tmp_sd = {}
    for n, m in model.named_modules():
        aiu_name = LUTorg_mod_to_aiu_name.get(n, None)
        if aiu_name in nodes_in_preccfg:
            assert (
                preccfg["precision"][f"{aiu_name}/precision"] == "int8"
            ), "Only support INT8 for now"
            tmp_sd[f"{n}.quantize_feature.clip_val"] = torch.Tensor(
                [preccfg["alpha_ap"][f"{aiu_name}/alpha"]]
            )
            tmp_sd[f"{n}.quantize_feature.clip_valn"] = torch.Tensor(
                [preccfg["alpha_an"][f"{aiu_name}/alpha"]]
            )
            tmp_sd[f"{n}.quantize_weight.clip_val"] = torch.Tensor(
                [preccfg["alpha_wp"][f"{aiu_name}/kernel"]]
            )
            w_scale = (
                preccfg["alpha_wp"][f"{aiu_name}/kernel"] / 127
            )  # this is a python float
            tmp_sd[f"{n}.weight"] = m.weight * w_scale  # dQ the INT W
            node_used.append(aiu_name)

    model.load_state_dict(tmp_sd, strict=False)
    del tmp_sd

    # --- Compare with model_ref, all clipvals should be the same
    if sim_level == 0:
        # if nothing to compare to, return the qmodel directly
        if model_ref is None:
            logger.warning(
                "Please provide a reference model if detailed comparison is needed."
            )
            return model.to(target_device)

    elif sim_level in [1, 2, 3]:
        # NOTE use the qmodel (lvl0) we just created (from precconf) as a reference
        #       use (backup'ed before qmodel_prep) intW model to create the lowered model
        if not model_ref:
            model_ref = model.to("cpu")
            model = model_intW_backup.to(target_device)

        if sim_level == 1:
            Qlin = QLinear_aiu_veri
            # will use torch.matmul as imatmul
        else:
            Qlin = QLinearINT8Deploy
            # will use real INT kernel

        for n, m in model.named_modules():
            aiu_name = LUTorg_mod_to_aiu_name.get(n, None)
            if aiu_name in nodes_in_preccfg and isinstance(m, torch.nn.Linear):
                parent_name, module_name = _parent_name(n)
                parent_mod = model.get_submodule(parent_name)
                additional_kwargs = {
                    "use_PT_native_Qfunc": qcfg.get("use_PT_native_Qfunc", False)
                }
                if sim_level == 3:
                    additional_kwargs.update(
                        {"chunk_size": 64, "max_acc_bits": 24, "truncate_lsb": 8}
                    )

                setattr(
                    parent_mod,
                    module_name,
                    Qlin.from_torch_iW(
                        m,
                        preccfg["precision"][f"{aiu_name}/precision"],
                        preccfg["alpha_ap"][f"{aiu_name}/alpha"],
                        preccfg["alpha_an"][f"{aiu_name}/alpha"],
                        preccfg["alpha_wp"][f"{aiu_name}/kernel"],
                        preccfg["zero_shift"].get(f"{aiu_name}/kernel", 0.0),
                        # if symmetric quantizer was used, zero_shift may not exist in precconfig
                        **additional_kwargs,
                    ),
                )
                # NOTE double check if zero_shift has overflow
                new_lin = getattr(parent_mod, module_name)
                if (
                    new_lin.zero_shift.max() > torch.finfo(torch.float16).max
                    or new_lin.zero_shift.min() < torch.finfo(torch.float16).min
                ):
                    logger.warning(f"Zero_shift may have overflow issue in {n}!")

                node_used.append(aiu_name)

    if compare_with_ref:
        compare_models(model, model_ref, sample_inp, target_device)

    unused = set(nodes_in_preccfg) - set(node_used)
    if len(unused) > 0:
        logger.info(f"unused nodes in precconfig {unused}")

    return model.to(target_device)


class QLinear_aiu_veri(torch.nn.Linear):
    """
    A QLinear class for AIU verification, no backward
    weight is stored in torch.int8, qparams are read from precconfig,
    including "correction term", i.e. "zero_shift" in precconfig
    also need to override forward to make it   Q->Linear->dQ
                                (as opposed to Q->dQ->Linear)
    """

    @classmethod
    def from_torch_iW(cls, nnlin_iW, prec, a_cv, a_cvn, w_cv, zero_shift, **kwargs):
        """
        TODO

        Args:
            nnlin_iW (_type_): _description_
            prec (_type_): _description_
            a_cv (_type_): _description_
            a_cvn (_type_): _description_
            w_cv (_type_): _description_
            zero_shift (_type_): _description_

        Returns:
            _type_: _description_
        """
        # NOTE prec, a_cv, a_cvn, ... are directly from precconfig.json,
        # whose dtypes are either python floats or strings
        assert prec == "int8", "Only support INT8 for now."
        # Standard

        target_device = kwargs.get(
            "target_device", kwargs.get("device", next(nnlin_iW.parameters()).device)
        )

        qlin_aiu = cls(
            nnlin_iW.in_features,
            nnlin_iW.out_features,
            bias=nnlin_iW.bias is not None,
            device=target_device,
        )

        qlin_aiu.nbits_a = 8  # only support INT8 for now
        qlin_aiu.nbits_w = 8
        qlin_aiu.acc_dtype = torch.float16  # aiu actually uses a fp16 variation
        qlin_aiu.usePTnativeQfunc = kwargs.get("use_PT_native_Qfunc", True)

        qlin_aiu.weight = nnlin_iW.weight  # already in INT, but saved as float
        with torch.no_grad():
            if qlin_aiu.usePTnativeQfunc:
                input_scale = torch.Tensor([(a_cv - a_cvn) / (2**qlin_aiu.nbits_a - 1)])
                input_zero_point = torch.round(-a_cvn / input_scale).to(torch.int)
                w_scale = torch.Tensor([w_cv * 2 / (2**qlin_aiu.nbits_w - 2)])
            else:
                # fms_mo formula is a bit different from conventional PT formula
                quant_scale = (2**qlin_aiu.nbits_a - 1) / torch.Tensor([a_cv - a_cvn])
                quant_stepsize = 1.0 / quant_scale
                quant_zero_point = torch.round(a_cvn * quant_scale)
                input_scale = quant_stepsize
                input_zero_point = -quant_zero_point
                quant_w_scale = (2**qlin_aiu.nbits_a - 2) / torch.Tensor([w_cv * 2])
                w_scale = 1.0 / quant_w_scale
                qlin_aiu.register_buffer("quant_scale", quant_scale)
                qlin_aiu.register_buffer("quant_stepsize", quant_stepsize)
                qlin_aiu.register_buffer("quant_zero_point", quant_zero_point)
            w_zp = torch.zeros_like(w_scale, dtype=torch.int)

            qlin_aiu.register_buffer("input_scale", input_scale)
            qlin_aiu.register_buffer("input_zp", input_zero_point)
            qlin_aiu.register_buffer("w_scale", w_scale)
            qlin_aiu.register_buffer("w_zp", w_zp)
            # store original cv_a and cv_w (in python floats, not tensors),
            # and sq scales... for later verification
            qlin_aiu.cvs = [a_cv, a_cvn, w_cv]

            if isinstance(zero_shift, str):
                zero_s = torch.Tensor(
                    json.loads(zero_shift)
                )  # , device=target_device )
            else:  # sym cases has no zero_shift
                zero_s = torch.Tensor([zero_shift])  # , device=target_device )
            corr_term = (
                (input_zero_point - 128) * zero_s * w_scale * input_scale
            )  # current AIU settings uses this sequence
            # zero_shift = intW.sum(dim=1), as w_int is [out,in], after sum -> [out,],
            # same as w_scale and bias.
            # NOTE: Use fp32 here to make sure dtype is large enough (as fp16 could overflow)
            qlin_aiu.register_buffer("corr_term", corr_term)  # [DEBUG only]
            qlin_aiu.register_buffer("zero_shift", zero_s)  # [DEBUG only]
            if nnlin_iW.bias is not None:
                qlin_aiu.bias = nnlin_iW.bias
                qlin_aiu.org_mod_has_bias = True
            else:
                qlin_aiu.org_mod_has_bias = False

        return qlin_aiu.to(target_device)

    def _get_name(self):
        return "QLinear_aiu_veri"

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bias={self.org_mod_has_bias}"

    def forward(
        self,
        input: torch.Tensor,  # pylint: disable=redefined-builtin
    ) -> torch.Tensor:
        with torch.no_grad():
            # Q, imatmul, add bias/corr, dQ, reshape should be all taken care of in the iaddmm
            # simplify to either real iaddmm or iadd_FP, one-liner here but graph will differ
            # NOTE to avoid confusion, imatmul should be like matmul, and self.W should stay
            # [out,in] which will need correct dims, i.e. [m,k]@[k,n], hence W.t()
            org_dtype = input.dtype
            re_shape = (-1, input.shape[-1])
            tar_shape = tuple(input.shape[:-1]) + (
                self.weight.shape[0],
            )  # remember W.shape=[out,in]

            # 1. Qa(x)
            if self.usePTnativeQfunc:
                input = torch.clamp(
                    (input / self.input_scale + self.input_zp - 128).round(), -128, 127
                )
            else:
                # fms_mo formula
                input = (
                    torch.round(
                        input.clamp(self.cvs[1], self.cvs[0]) / self.quant_stepsize
                        - self.quant_zero_point
                    )
                    - 128
                )

            # 2. imatmul, should output "INT32" on HW, should have no overflow
            # 3. dQ, and add bias/correction_term
            # for example:
            # x = torch.matmul(x.view(re_shape), self.weight.t())
            # x = (x - (self.input_zp-128)*self.zero_shift)*self.w_scale.to(self.acc_dtype)*\
            #   self.input_scale.to(self.acc_dtype) + self.bias.to(self.acc_dtype)

            # OR 2+3 into one torch.addmm step (input, mat1, mat2),   out=β input + α (mat1@mat2)
            # https://pytorch.org/docs/stable/generated/torch.addmm.html,
            out_tmp = torch.zeros(
                (input.shape[0] * input.shape[1], self.weight.shape[0]),
                dtype=self.acc_dtype,
                device=input.device,
            )
            torch.addmm(
                (self.bias - self.corr_term).to(
                    self.acc_dtype
                ),  # we could pre-calc this in init, this is just for clarity and debug
                mat1=input.view(re_shape).to(
                    self.acc_dtype
                ),  # real imatmul requires (2d tensor x 2d tensor), hence the reshape
                mat2=self.weight.t().to(self.acc_dtype),
                alpha=(self.w_scale * self.input_scale)
                .to(self.acc_dtype)
                .item(),  # assume both W and A are per-T
                beta=1.0,
                out=out_tmp,
            )
            # 4. reshape
            input = out_tmp.reshape(tar_shape).to(org_dtype)
        return input
