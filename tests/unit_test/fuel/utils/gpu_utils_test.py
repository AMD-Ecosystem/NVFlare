# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
from unittest.mock import patch

import pytest

from nvflare.fuel.utils import gpu_utils

_MIB = 1024 * 1024

# Captured real rocm-smi CSV fixtures (ROCm 10.x). VRAM is in BYTES per card;
# the shim converts to MiB (nvidia-smi contract) and computes free = total - used.
ROCM_SHOWID_CSV = [
    "device,Device Name,Device ID,Device Rev,Subsystem ID,GUID",
    "card0,AMD Instinct MI300X,0x74a1,0x00,0x74a1,28851",
    "card1,AMD Instinct MI300X,0x74a1,0x00,0x74a1,28852",
]

ROCM_MEMINFO_CSV = [
    "device,VRAM Total Memory (B),VRAM Total Used Memory (B)",
    "card0,206141652992,464486400",
    "card1,206141652992,1052266496",
]

# Same data but with the Used column emitted BEFORE the Total column -- exercises
# the header-driven, order-tolerant, exclusive total/used column matching.
ROCM_MEMINFO_CSV_REORDERED = [
    "device,VRAM Total Used Memory (B),VRAM Total Memory (B)",
    "card0,464486400,206141652992",
    "card1,1052266496,206141652992",
]

# 206141652992 B // (1024*1024) = 196592 MiB (per card).
EXPECTED_TOTAL_MIB = [196592, 196592]
# free = (total - used) // MiB.
EXPECTED_FREE_MIB = [
    (206141652992 - 464486400) // _MIB,  # 196149
    (206141652992 - 1052266496) // _MIB,  # 195568
]

NVIDIA_INDEX_CSV = ["index", "0", "1"]
NVIDIA_MEM_TOTAL_CSV = ["memory.total [MiB]", "40960 MiB", "40960 MiB"]
NVIDIA_MEM_FREE_CSV = ["memory.free [MiB]", "40000 MiB", "39000 MiB"]


class TestNvidiaPathUnchanged:
    """When nvidia-smi is present the NVIDIA path is used unchanged."""

    @patch.object(gpu_utils, "has_nvidia_smi", return_value=True)
    @patch.object(gpu_utils, "use_nvidia_smi", return_value=NVIDIA_INDEX_CSV)
    def test_get_host_gpu_ids_nvidia(self, mock_smi, mock_has):
        assert gpu_utils.get_host_gpu_ids() == [0, 1]

    @patch.object(gpu_utils, "has_nvidia_smi", return_value=True)
    @patch.object(gpu_utils, "use_nvidia_smi", return_value=NVIDIA_MEM_TOTAL_CSV)
    def test_get_host_gpu_memory_total_nvidia(self, mock_smi, mock_has):
        assert gpu_utils.get_host_gpu_memory_total() == [40960, 40960]

    @patch.object(gpu_utils, "has_nvidia_smi", return_value=True)
    @patch.object(gpu_utils, "use_nvidia_smi", return_value=NVIDIA_MEM_FREE_CSV)
    def test_get_host_gpu_memory_free_nvidia(self, mock_smi, mock_has):
        assert gpu_utils.get_host_gpu_memory_free() == [40000, 39000]


class TestNoVendorFallback:
    """With neither nvidia-smi nor rocm-smi present, all queries return []."""

    @patch.object(gpu_utils, "has_nvidia_smi", return_value=False)
    @patch.object(gpu_utils, "has_rocm_smi", return_value=False)
    def test_ids_empty_when_no_vendor(self, mock_rocm, mock_nvidia):
        assert gpu_utils.get_host_gpu_ids() == []

    @patch.object(gpu_utils, "has_nvidia_smi", return_value=False)
    @patch.object(gpu_utils, "has_rocm_smi", return_value=False)
    def test_total_empty_when_no_vendor(self, mock_rocm, mock_nvidia):
        assert gpu_utils.get_host_gpu_memory_total() == []

    @patch.object(gpu_utils, "has_nvidia_smi", return_value=False)
    @patch.object(gpu_utils, "has_rocm_smi", return_value=False)
    def test_free_empty_when_no_vendor(self, mock_rocm, mock_nvidia):
        assert gpu_utils.get_host_gpu_memory_free() == []


class TestRocmPath:
    """When only rocm-smi is present the AMD path enumerates indices + MiB."""

    @patch.object(gpu_utils, "has_nvidia_smi", return_value=False)
    @patch.object(gpu_utils, "has_rocm_smi", return_value=True)
    @patch.object(gpu_utils, "use_rocm_smi", return_value=ROCM_SHOWID_CSV)
    def test_get_host_gpu_ids_rocm(self, mock_smi, mock_rocm, mock_nvidia):
        assert gpu_utils.get_host_gpu_ids() == [0, 1]

    @patch.object(gpu_utils, "has_nvidia_smi", return_value=False)
    @patch.object(gpu_utils, "has_rocm_smi", return_value=True)
    @patch.object(gpu_utils, "use_rocm_smi", return_value=ROCM_MEMINFO_CSV)
    def test_get_host_gpu_memory_total_rocm_bytes_to_mib(self, mock_smi, mock_rocm, mock_nvidia):
        assert gpu_utils.get_host_gpu_memory_total() == EXPECTED_TOTAL_MIB

    @patch.object(gpu_utils, "has_nvidia_smi", return_value=False)
    @patch.object(gpu_utils, "has_rocm_smi", return_value=True)
    @patch.object(gpu_utils, "use_rocm_smi", return_value=ROCM_MEMINFO_CSV_REORDERED)
    def test_get_host_gpu_memory_free_rocm_total_minus_used(self, mock_smi, mock_rocm, mock_nvidia):
        # Column order is reversed (Used before Total): exclusive matching must
        # still resolve total=Total-column, used=Used-column so free is correct.
        assert gpu_utils.get_host_gpu_memory_free() == EXPECTED_FREE_MIB

    @patch.object(gpu_utils, "has_nvidia_smi", return_value=False)
    @patch.object(gpu_utils, "has_rocm_smi", return_value=True)
    def test_ids_fallback_to_meminfo_when_showid_empty(self, mock_rocm, mock_nvidia):
        # --showid yields no card rows -> fall back to the meminfo card rows.
        def _fake(args):
            if "--showid" in args:
                return ["device,Device Name"]  # header only, no cards
            return ROCM_MEMINFO_CSV

        with patch.object(gpu_utils, "use_rocm_smi", side_effect=_fake):
            assert gpu_utils.get_host_gpu_ids() == [0, 1]

    @patch.object(gpu_utils, "has_nvidia_smi", return_value=False)
    @patch.object(gpu_utils, "has_rocm_smi", return_value=True)
    @patch.object(gpu_utils, "use_rocm_smi", return_value=ROCM_MEMINFO_CSV)
    def test_rocm_unit_must_be_mib(self, mock_smi, mock_rocm, mock_nvidia):
        with pytest.raises(RuntimeError):
            gpu_utils.get_host_gpu_memory_total(unit="GiB")


class TestVendorPreference:
    """When both vendors are present, nvidia-smi is preferred; rocm-smi is not consulted."""

    @patch.object(gpu_utils, "has_nvidia_smi", return_value=True)
    @patch.object(gpu_utils, "has_rocm_smi", return_value=True)
    @patch.object(gpu_utils, "use_nvidia_smi", return_value=NVIDIA_INDEX_CSV)
    @patch.object(gpu_utils, "use_rocm_smi")
    def test_nvidia_preferred_when_both_present(self, mock_rocm_smi, mock_nvidia_smi, mock_has_rocm, mock_has_nvidia):
        assert gpu_utils.get_host_gpu_ids() == [0, 1]
        mock_rocm_smi.assert_not_called()
