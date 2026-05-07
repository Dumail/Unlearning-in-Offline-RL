"""
download_data → build_negative_set → train_base_dt(tiny) → evaluate_tmi
→ run_unlearning(ascent=2, refit=2) → analyze_b1_b4_multiseed

uv run python scripts/run_release_smoke.py --env hopper_mr --seed 0
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_FULL_MAP = {
    "hopper_mr": "hopper-medium-replay-v2",
    "halfcheetah_mr": "halfcheetah-medium-replay-v2",
    "walker2d_mr": "walker2d-medium-replay-v2",
    "hopper_m": "hopper-medium-v2",
    "halfcheetah_m": "halfcheetah-medium-v2",
    "walker2d_m": "walker2d-medium-v2",
    "hopper_me": "hopper-medium-expert-v2",
    "halfcheetah_me": "halfcheetah-medium-expert-v2",
    "walker2d_me": "walker2d-medium-expert-v2",
}


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}")
    rc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False).returncode
    if rc != 0:
        raise SystemExit(f"Smoke failed: command exited with code {rc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Release end-to-end smoke test")
    parser.add_argument("--env", default="hopper_mr", choices=list(ENV_FULL_MAP))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip download_data + build_negative_set if already done, to speed up the test.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env_short = args.env
    env_full = ENV_FULL_MAP[env_short]
    seed = args.seed

    if not args.skip_data_prep:
        run(
            [
                "uv",
                "run",
                "python",
                "scripts/download_data.py",
                f"env={env_short}",
            ]
        )
        run(
            [
                "uv",
                "run",
                "python",
                "scripts/build_negative_set.py",
                f"env={env_short}",
            ]
        )

    run(
        [
            "uv",
            "run",
            "python",
            "scripts/train_base_dt.py",
            f"env={env_short}",
            f"seed={seed}",
            "train=tiny",
        ]
    )
    run(
        [
            "uv",
            "run",
            "python",
            "scripts/evaluate_tmi.py",
            f"env={env_short}",
            f"seed={seed}",
        ]
    )
    run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_unlearning.py",
            f"env={env_short}",
            f"seed={seed}",
            "train=tiny",
            "unlearn.kl_weight=0.1",
            "unlearn.ascent_steps=2",
            "unlearn.refit_steps=2",
        ]
    )

    results_env = PROJECT_ROOT / "results" / env_full
    must_exist = [
        results_env / "tmi_eval_dt_final.json",
        results_env / f"ga_refit_lambda0.1_steps2_seed{seed}.json",
    ]
    for path in must_exist:
        if not path.exists():
            raise SystemExit(f"Smoke failed: did not generate {path}")
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Smoke failed: {path} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit(f"Smoke failed: {path} top-level is not a dict")
        if "forget_auc" not in payload:
            raise SystemExit(f"Smoke failed: {path} missing forget_auc field")

    print("\n[smoke] All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
