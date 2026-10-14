import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# transformers >= 5.18 resolves Qwen3.5's Gated DeltaNet to fla's Triton kernels when `fla` is importable - at import
# time, with no device check (use_kernel_func_from_hub_with_fallback). Everything the tests build lives on the CPU, so
# with fla-core installed (the server since 2026-10-14) Triton raised "Pointer argument cannot be accessed from Triton"
# across the suite (p0 on run440212). The suite tests the torch reference path; LIVE3R_TEST_FLA=1 lets fla through.
# This must run before any test module imports transformers' models.
if os.environ.get("LIVE3R_TEST_FLA") != "1":
    sys.modules.setdefault("fla", None)
