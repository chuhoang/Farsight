"""Import CVLface's `aligners` / `models` packages (third_party, run_v1) under unique names.
Generic top-level names like `models` collide with other repos (AIM, BPJDet), so we never put run_v1 on sys.path."""
import importlib.util
import sys
import types

from farsight.core.weights import ROOT

RUN_V1 = ROOT / "third_party" / "CVLface" / "cvlface" / "research" / "recognition" / "code" / "run_v1"


def load_pkg(name):
    alias = f"_cvlface_{name}"
    if alias in sys.modules:
        return sys.modules[alias]
    if name == "models" and importlib.util.find_spec("rpe_index_cpp") is None:
        # rpe_ops is an optional CUDA ext (build: models/vit_kprpe/RPE/rpe_ops/setup.py, ~10x less memory in
        # training); on ImportError CVLface runs `setup.py install` (and sys.exit()s on success). Without the
        # built ext, pre-register a stub so it takes its pure-torch fallback path instead.
        stub = types.ModuleType(f"{alias}.vit_kprpe.RPE.rpe_ops.rpe_index")
        stub.RPEIndexFunction = None
        sys.modules[stub.__name__] = stub
    spec = importlib.util.spec_from_file_location(
        alias, RUN_V1 / name / "__init__.py", submodule_search_locations=[str(RUN_V1 / name)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod
