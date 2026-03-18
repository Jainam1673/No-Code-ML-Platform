from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PlatformCapabilities:
    python_version: str
    os_name: str
    arch: str
    cuda_available: bool
    nvidia_smi_available: bool
    gpu_count: int
    accelerator_hint: str


class CapabilityService:
    """Collects runtime platform capabilities for operations and scheduling."""

    def detect(self) -> PlatformCapabilities:
        nvidia_smi = shutil.which("nvidia-smi") is not None
        gpu_count = self._detect_gpu_count() if nvidia_smi else 0
        cuda_available = gpu_count > 0

        accelerator_hint = "cpu"
        if cuda_available:
            accelerator_hint = "gpu"
        elif os.getenv("TPU_NAME") or os.getenv("CLOUD_TPU_TASK_ID"):
            accelerator_hint = "tpu"

        return PlatformCapabilities(
            python_version=platform.python_version(),
            os_name=platform.system().lower(),
            arch=platform.machine().lower(),
            cuda_available=cuda_available,
            nvidia_smi_available=nvidia_smi,
            gpu_count=gpu_count,
            accelerator_hint=accelerator_hint,
        )

    @staticmethod
    def _detect_gpu_count() -> int:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            return len(lines)
        except Exception:  # noqa: BLE001
            return 0


capability_service = CapabilityService()
