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

"""Unit tests for run_quant.py"""

# Standard
import copy
import json
import os

# Third Party
import pytest
import torch

# Local
from fms_mo.run_quant import get_parser, parse_arguments, quantize
from fms_mo.training_args import (
    DataArguments,
    FMSMOArguments,
    GPTQArguments,
    ModelArguments,
    OptArguments,
)
from tests.artifacts.testdata import MODEL_NAME, WIKITEXT_TOKENIZED_DATA_JSON

MODEL_ARGS = ModelArguments(model_name_or_path=MODEL_NAME, torch_dtype="float16")
DATA_ARGS = DataArguments(
    training_data_path=WIKITEXT_TOKENIZED_DATA_JSON,
)
OPT_ARGS = OptArguments(quant_method="dq", output_dir="tmp")
GPTQ_ARGS = GPTQArguments(
    bits=4,
    group_size=64,
)
DQ_ARGS = FMSMOArguments(
    nbits_w=8,
    nbits_a=8,
    nbits_kvcache=32,
    qa_mode="fp8_e4m3_scale",
    qw_mode="fp8_e4m3_scale",
    qmodel_calibration_new=0,
)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Only runs if GPUs are available"
)
def test_run_train_requires_output_dir():
    """Check fails when output dir not provided."""
    updated_output_dir_opt_args = copy.deepcopy(OPT_ARGS)
    updated_output_dir_opt_args.output_dir = None
    with pytest.raises(TypeError):
        quantize(
            model_args=MODEL_ARGS,
            data_args=DATA_ARGS,
            opt_args=updated_output_dir_opt_args,
            fms_mo_args=DQ_ARGS,
        )


def test_run_train_fails_training_data_path_not_exist():
    """Check fails when data path not found."""
    updated_data_path_args = copy.deepcopy(DATA_ARGS)
    updated_data_path_args.training_data_path = "fake/path"
    with pytest.raises(FileNotFoundError):
        quantize(
            model_args=MODEL_ARGS,
            data_args=updated_data_path_args,
            opt_args=OPT_ARGS,
            fms_mo_args=DQ_ARGS,
        )


HAPPY_PATH_DUMMY_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "artifacts", "configs", "dummy_job_config.json"
)


@pytest.fixture(name="job_config", scope="session")
def fixture_job_config():
    """Fixture to get happy path dummy config as a dict, note that job_config dict gets
    modified during process training args"""
    with open(HAPPY_PATH_DUMMY_CONFIG_PATH, "r", encoding="utf-8") as f:
        dummy_job_config_dict = json.load(f)
    return dummy_job_config_dict


############################# Arg Parsing Tests #############################


def test_parse_arguments(job_config):
    """Test that arg parser can parse json job config correctly"""
    parser = get_parser()
    job_config_copy = copy.deepcopy(job_config)
    (
        model_args,
        data_args,
        opt_args,
        _,
        _,
        _,
    ) = parse_arguments(parser, job_config_copy)
    assert str(model_args.torch_dtype) == "torch.bfloat16"
    assert data_args.training_data_path == "data_train"
    assert opt_args.output_dir == "models/Maykeye/TinyLLama-v0-GPTQ"
    assert opt_args.quant_method == "gptq"


def test_parse_arguments_defaults(job_config):
    """Test that defaults set in fms_mo/training_args.py are retained"""
    parser = get_parser()
    job_config_defaults = copy.deepcopy(job_config)
    assert "torch_dtype" not in job_config_defaults
    assert "max_seq_length" not in job_config_defaults
    assert "model_revision" not in job_config_defaults
    assert "nbits_kvcache" not in job_config_defaults
    (
        model_args,
        data_args,
        _,
        fms_mo_args,
        _,
        _,
    ) = parse_arguments(parser, job_config_defaults)
    assert str(model_args.torch_dtype) == "torch.bfloat16"
    assert model_args.model_revision == "main"
    assert data_args.max_seq_length == 2048
    assert fms_mo_args.nbits_kvcache == 32
