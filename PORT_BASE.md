# nvflare ROCm port -- base + application

This is a framework-transparent enablement port. The FULL ported tree is:

    upstream NVIDIA/NVFlare @ a700b64b3654  ("Disable inbound admin services in Simulator (#5125)")
    + the two files in this snapshot applied at their real repo paths:
        nvflare/fuel/utils/gpu_utils.py               (rocm-smi GPU-enumeration path)
        tests/unit_test/fuel/utils/gpu_utils_test.py  (12 fixture-driven unit tests)

Reconstruct/validate with hipshift/repro/gfx942_unit.sh (or gfx950_unit.sh):
those clone upstream at the base and re-apply these two files inside the ROCm-10
manylinux container, regenerating the golden enumeration/parity/unit/coverage logs.

NOTE: the full upstream tree is intentionally NOT snapshotted here because upstream
NVFlare ships test-fixture PEM private keys (nvflare/tool/cert/cert_commands.py,
tests/unit_test/fuel/utils/secret_utils_test.py, etc.) that trip GitHub push
protection on the control-plane repo. Those are upstream fixtures, not AMD secrets.
The 2 AMD files + this base pin are the complete, durable AMD contribution.
