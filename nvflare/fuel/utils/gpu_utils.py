# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import subprocess
from typing import List

_MIB = 1024 * 1024


def has_nvidia_smi() -> bool:
    from shutil import which

    return which("nvidia-smi") is not None


def has_rocm_smi() -> bool:
    from shutil import which

    return which("rocm-smi") is not None


def use_nvidia_smi(query: str, report_format: str = "csv"):
    if has_nvidia_smi():
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", f"--format={report_format}"],
            capture_output=True,
            text=True,
        )
        rc = result.returncode
        if rc > 0:
            raise Exception(f"Failed to call nvidia-smi with query {query}", result.stderr)
        else:
            return result.stdout.splitlines()
    return None


def use_rocm_smi(args: List[str]):
    """Runs rocm-smi with the given arguments and returns stdout lines.

    Returns None if rocm-smi is not present. Raises on a non-zero return code so
    the caller can surface the failure the same way the nvidia-smi path does.
    """
    if has_rocm_smi():
        result = subprocess.run(
            ["rocm-smi", *args],
            capture_output=True,
            text=True,
        )
        rc = result.returncode
        if rc > 0:
            raise Exception(f"Failed to call rocm-smi with args {args}", result.stderr)
        return result.stdout.splitlines()
    return None


def _parse_gpu_mem(result: str = None, unit: str = "MiB") -> List:
    gpu_memory = []
    if result:
        for i in result[1:]:
            mem, mem_unit = i.split(" ")
            if mem_unit != unit:
                raise RuntimeError("Memory unit does not match.")
            gpu_memory.append(int(mem))
    return gpu_memory


def _rocm_csv_rows(lines: List[str]):
    """Yields (header, data-rows) from rocm-smi --csv output.

    rocm-smi emits a header row followed by one row per card (``card0``,
    ``card1``, ...). Blank/summary trailer lines are ignored. Returns
    ``(header_cols, [row_cols, ...])`` or ``(None, [])`` when there is no data.
    """
    if not lines:
        return None, []
    rows = [ln.strip() for ln in lines if ln.strip()]
    if len(rows) < 2:
        return None, []
    header = [c.strip() for c in rows[0].split(",")]
    data = []
    for ln in rows[1:]:
        cols = [c.strip() for c in ln.split(",")]
        # Only real card rows (the device column starts with "card").
        if cols and cols[0].lower().startswith("card"):
            data.append(cols)
    return header, data


def _rocm_mem_col(header: List[str], kw: str) -> int:
    """Finds the index of the VRAM ``total`` or ``used`` column, header-driven.

    Tolerates rocm-smi column-name/order drift across ROCm versions. The "total"
    column MUST be exclusive of the Used column: "VRAM Total Used Memory (B)"
    contains BOTH "total" and "used", so matching "total" alone would collide
    with the Used column if rocm-smi ever emits Used before Total. So for the
    total column we require "total" present AND "used" absent; the used column
    (only the Used column contains "used") needs no exclusion.
    """
    exclude = "used" if kw == "total" else None
    for idx, name in enumerate(header):
        low = name.lower()
        if "vram" in low and kw in low and (exclude is None or exclude not in low):
            return idx
    raise RuntimeError(f"rocm-smi VRAM '{kw}' column not found in header: {header}")


def _rocm_gpu_ids() -> List:
    """Enumerates AMD GPU indices via rocm-smi.

    Uses ``rocm-smi --showid --csv``; the device column is ``card<N>`` and N is
    the integer index. Falls back to ``--showmeminfo vram --csv`` (also one row
    per card) if --showid yields no card rows.
    """
    header, data = _rocm_csv_rows(use_rocm_smi(["--showid", "--csv"]) or [])
    if not data:
        # Fallback: derive indices from the meminfo card rows.
        header, data = _rocm_csv_rows(use_rocm_smi(["--showmeminfo", "vram", "--csv"]) or [])
    ids = []
    for cols in data:
        dev = cols[0].lower()
        # "card0" -> 0
        num = "".join(ch for ch in dev if ch.isdigit())
        if num != "":
            ids.append(int(num))
    return ids


def _rocm_gpu_mem(unit="MiB", which: str = "total") -> List:
    """Enumerates per-card VRAM in the requested unit via rocm-smi.

    rocm-smi VRAM is reported in BYTES per card; we convert to MiB to match the
    nvidia-smi MiB contract. rocm-smi has no native free query, so
    ``free = total - used``.
    """
    if unit != "MiB":
        raise RuntimeError("rocm-smi enumeration only supports the MiB unit.")
    lines = use_rocm_smi(["--showmeminfo", "vram", "--csv"]) or []
    header, data = _rocm_csv_rows(lines)
    if not data:
        return []
    total_idx = _rocm_mem_col(header, "total")
    used_idx = _rocm_mem_col(header, "used")
    mem = []
    for cols in data:
        total_b = int(cols[total_idx])
        used_b = int(cols[used_idx])
        if which == "total":
            mem.append(total_b // _MIB)
        else:  # free = total - used
            mem.append((total_b - used_b) // _MIB)
    return mem


def get_host_gpu_memory_total(unit="MiB") -> List:
    if has_nvidia_smi():
        result = use_nvidia_smi("memory.total")
        return _parse_gpu_mem(result, unit)
    if has_rocm_smi():
        return _rocm_gpu_mem(unit, which="total")
    return []


def get_host_gpu_memory_free(unit="MiB") -> List:
    if has_nvidia_smi():
        result = use_nvidia_smi("memory.free")
        return _parse_gpu_mem(result, unit)
    if has_rocm_smi():
        return _rocm_gpu_mem(unit, which="free")
    return []


def get_host_gpu_ids() -> List:
    """Gets GPU IDs.

    Note:
        Supports nvidia-smi (NVIDIA) and rocm-smi (AMD). When nvidia-smi is
        present it is always preferred (behavior unchanged); rocm-smi is used
        only when nvidia-smi is absent; when neither vendor tool is present an
        empty list is returned (unchanged graceful fallback).
    """
    if has_nvidia_smi():
        result = use_nvidia_smi("index")
        gpu_ids = []
        if result:
            for i in result[1:]:
                gpu_ids.append(int(i))
        return gpu_ids
    if has_rocm_smi():
        return _rocm_gpu_ids()
    return []
