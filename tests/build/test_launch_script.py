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

"""Unit Tests for accelerate_launch script.
"""

# Standard
import os
import tempfile
import glob

# Third Party
import pytest

# First Party
from build.accelerate_launch import main
from build.utils import serialize_args
from tests.artifacts.testdata import WIKITEXT_TOKENIZED_DATA_JSON
from fms_mo.utils.error_logging import (
    USER_ERROR_EXIT_CODE,
    INTERNAL_ERROR_EXIT_CODE,
)
from fms_mo.utils.import_utils import available_packages


SCRIPT = "fms_mo/run_quant.py"
MODEL_NAME = "Maykeye/TinyLLama-v0"
BASE_KWARGS = {
    "model_name_or_path": MODEL_NAME,
    "output_dir": "tmp",
}
BASE_GPTQ_KWARGS = {
    **BASE_KWARGS,
    **{
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
        "training_data_path": WIKITEXT_TOKENIZED_DATA_JSON,
    },
}
BASE_FP8_KWARGS = {
    **BASE_KWARGS,
    **{
        "quant_method": "fp8",
    },
}


def setup_env(tempdir):
    os.environ["TRAINING_SCRIPT"] = SCRIPT
    os.environ["PYTHONPATH"] = "./:$PYTHONPATH"
    os.environ["TERMINATION_LOG_FILE"] = tempdir + "/termination-log"


def cleanup_env():
    os.environ.pop("OPTIMIZER_SCRIPT", None)
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("TERMINATION_LOG_FILE", None)

### Tests for model dtype edge cases
@pytest.mark.skipif(not available_packages["auto_gptq"], reason="Only runs if auto-gptq package is installed")
def test_successful_gptq():
    """Check if we can gptq models"""
    with tempfile.TemporaryDirectory() as tempdir:
        setup_env(tempdir)
        QUANT_KWARGS = {**BASE_KWARGS, **{"output_dir": tempdir}}
        serialized_args = serialize_args(QUANT_KWARGS)
        os.environ["FMS_MO_CONFIG_JSON_ENV_VAR"] = serialized_args

        assert main() == 0