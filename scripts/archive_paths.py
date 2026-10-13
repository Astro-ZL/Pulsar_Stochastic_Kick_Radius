from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MCMC = RESULTS / "mcmc"
EVIDENCE = RESULTS / "evidence"
RUN_ROOT = Path(os.environ.get("PULSAR_RUN_DIR", str(ROOT / "reproduced"))).expanduser().resolve()
def run_dir(name):
    p = RUN_ROOT / name
    p.mkdir(parents=True, exist_ok=True)
    return p
