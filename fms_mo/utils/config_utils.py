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
import base64
import json
import os
import pickle


def update_config(config, **kwargs):
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warm user
                        print(f"Warning: {config_name} does not accept parameter: {k}")


def get_json_config():
    """Parses JSON configuration if provided via environment variables
    FMS_MO_CONFIG_JSON_ENV_VAR or FMS_MO_CONFIG_JSON_PATH.

    FMS_MO_CONFIG_JSON_ENV_VAR is the base64 encoded JSON.
    FMS_MO_CONFIG_JSON_PATH is the path to the JSON config file.

    Returns: dict or {}
    """
    json_env_var = os.getenv("FMS_MO_CONFIG_JSON_ENV_VAR")
    json_path = os.getenv("FMS_MO_CONFIG_JSON_PATH")

    # accepts either path to JSON file or encoded string config
    # env var takes precedent
    job_config_dict = {}
    if json_env_var:
        job_config_dict = txt_to_obj(json_env_var)
    elif json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            job_config_dict = json.load(f)

    return job_config_dict


def txt_to_obj(txt):
    """Given encoded byte string, converts to base64 decoded dict.

    Args:
        txt: str
    Returns: dict[str, Any]
    """
    base64_bytes = txt.encode("ascii")
    message_bytes = base64.b64decode(base64_bytes)
    try:
        # If the bytes represent JSON string
        return json.loads(message_bytes)
    except UnicodeDecodeError:
        # Otherwise the bytes are a pickled python dictionary
        return pickle.loads(message_bytes)
