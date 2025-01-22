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

# Standard
import copy
import json
import os
from unittest.mock import patch

# Third Party
import pytest

# Local
from build.utils import process_accelerate_launch_args

HAPPY_PATH_DUMMY_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "configs", "dummy_job_config.json"
)


# Note: job_config dict gets modified during processing training args
@pytest.fixture(name="job_config", scope="session")
def fixture_job_config():
    """Fixture to get happy path json job config as dict"""
    with open(HAPPY_PATH_DUMMY_CONFIG_PATH, "r", encoding="utf-8") as f:
        dummy_job_config_dict = json.load(f)
    return dummy_job_config_dict


def test_process_accelerate_launch_args(job_config):
    """Test to verify accelerate launch args can be parsed successfully"""
    args = process_accelerate_launch_args(job_config)
    # json config values passed in through job config
    assert args.main_process_port == 1234
    assert args.training_script == "fms_mo.run_quant"

    # default values
    assert args.tpu_use_cluster is False
    assert args.mixed_precision is None


@patch("torch.cuda.device_count", return_value=1)
def test_accelerate_launch_args_user_set_num_processes_ignored(job_config):
    """Test to verify that user specified num_processes is ignored if number of available
    GPUs is different"""
    job_config_copy = copy.deepcopy(job_config)
    job_config_copy["accelerate_launch_args"]["num_processes"] = "3"
    args = process_accelerate_launch_args(job_config_copy)
    # determine number of processes by number of GPUs available
    assert args.num_processes == 1

    # if single-gpu, CUDA_VISIBLE_DEVICES set
    assert os.getenv("CUDA_VISIBLE_DEVICES") == "0"


@patch.dict(os.environ, {"SET_NUM_PROCESSES_TO_NUM_GPUS": "False"})
def test_accelerate_launch_args_user_set_num_processes(job_config):
    """Test to verify user specified num_processes is used if SET_NUM_PROCESSES_TO_NUM_GPUS
    env var is disabled"""
    job_config_copy = copy.deepcopy(job_config)
    job_config_copy["accelerate_launch_args"]["num_processes"] = "3"

    args = process_accelerate_launch_args(job_config_copy)
    # json config values used
    assert args.num_processes == 3
