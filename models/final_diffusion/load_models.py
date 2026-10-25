"""Load either local checkpoint with its original initialization chain."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_model(kind="jev", device=None, dtype=None):
    """kind='jev' or 'diffusion_only'; returns (model, tokenizer, processor).

    Uses the shared local backbone and original checkpoint metadata.
    Call during startup: initialization temporarily changes the process cwd.
    """
    if kind not in ("jev", "diffusion_only"):
        raise ValueError("kind must be jev or diffusion_only")
    repo = str(ROOT.parents[1])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from djev.io import load
    previous_cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        return load(str(ROOT / "backbone"), str(ROOT / kind),
                    device=device, dtype=dtype)
    finally:
        os.chdir(previous_cwd)
