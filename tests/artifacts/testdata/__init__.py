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

"""Helpful datasets for configuring individual unit tests."""

# Standard
import os

### Constants used for data
MODEL_NAME = "Maykeye/TinyLLama-v0"
DATA_DIR = os.path.join(os.path.dirname(__file__))

WIKITEXT_TOKENIZED_DATA_JSON = os.path.join(
    DATA_DIR, "wiki_maykeye_tinyllama_v0_numsamp2_seqlen2048"
)
