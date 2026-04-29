# Copyright 2026 Jason Hurt
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
import torch
import sys
from pathlib import Path

lib_name = "attention_mps_torch_lib.so"

def _find_library():
    local_path = Path(__file__).parent / lib_name
    if local_path.exists():
        return local_path

    for path in sys.path:
        if "site-packages" in path:
            venv_path = Path(path) / "attention_mps" / lib_name
            if venv_path.exists():
                return venv_path

    return None


_lib_path = _find_library()

if _lib_path:
    torch.ops.load_library(str(_lib_path))
else:
    raise ImportError(f"Could not find {lib_name}. I checked the local folder and your venv site-packages.")
