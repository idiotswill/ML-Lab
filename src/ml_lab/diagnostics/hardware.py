from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    os: str
    os_version: str
    architecture: str
    cpu: str
    logical_cpus: int
    ram_bytes: int | None
    disk_total_bytes: int
    disk_free_bytes: int
    nvidia_gpu: str | None
    nvidia_vram_mb: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def collect_hardware_info(workspace_path: Path) -> HardwareInfo:
    total, _, free = shutil.disk_usage(workspace_path)
    gpu_name, gpu_vram = _nvidia_info()
    return HardwareInfo(
        os=platform.system(),
        os_version=platform.version(),
        architecture=platform.machine(),
        cpu=platform.processor() or platform.uname().processor or "Unknown CPU",
        logical_cpus=os.cpu_count() or 1,
        ram_bytes=_total_memory(),
        disk_total_bytes=total,
        disk_free_bytes=free,
        nvidia_gpu=gpu_name,
        nvidia_vram_mb=gpu_vram,
    )


def _total_memory() -> int | None:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * pages)
    except (AttributeError, ValueError, OSError):
        return None


def _nvidia_info() -> tuple[str | None, int | None]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None, None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None, None
    first = completed.stdout.splitlines()[0]
    try:
        name, memory = [part.strip() for part in first.rsplit(",", 1)]
        return name, int(memory)
    except (ValueError, TypeError):
        return first.strip() or None, None
