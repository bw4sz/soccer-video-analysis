"""Fetch recokick ball-action-spotting pretrained weights by file ID.

`gdown --folder` on the author's Drive folder repeatedly trips Google's
per-account daily download quota because it enumerates the huge cached-prediction
subtrees first. This script instead downloads each weight file by its individual
Drive ID (single-file mode, which is more quota-friendly) and **skips files that
already exist**, so it is safe to re-run after the ~24h quota reset.

Drive folder: https://drive.google.com/drive/folders/1mIu62cIdsRn3W4o1E5vRR8V5Q1B6HHoz

Usage:
    python scripts/bas_fetch_weights.py                 # all bundles
    python scripts/bas_fetch_weights.py --only ball_tuning_001
    python scripts/bas_fetch_weights.py --weights-only   # skip config/source/logs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEST = REPO / "vendor/ball-action-spotting/data/gdrive_weights/ball_action/experiments"

# id -> relative path under DEST. Enumerated via gdown.download_folder(skip_download=True).
FILES: dict[str, str] = {
    # ball_tuning_001 — final transfer-learned ball model (2-class: PASS, DRIVE)
    "14DcG4IhAhoSKATcUR6PMVMNDrKxr4cWE": "ball_tuning_001/fold_0/model-034-0.858899.pth",
    "1Aie3ZdqjGiu-ymYfrpL0WICrZLY5AMtU": "ball_tuning_001/fold_1/model-034-0.867644.pth",
    "1MQLVoBJ3xU6YSCOv2uAkzAZlm6U5bQxa": "ball_tuning_001/fold_2/model-034-0.748727.pth",
    "16vLiUtKuKdKANrFZo0mA4iRYIGFxqSDA": "ball_tuning_001/fold_3/model-034-0.877509.pth",
    "1Tynn_RkyQcgcdqiLGqS7LAKVieEazKiZ": "ball_tuning_001/fold_4/model-034-0.866331.pth",
    "1vf68yFgmUCYKNpTt1uDj3b4WcKrC8veb": "ball_tuning_001/fold_5/model-034-0.887485.pth",
    "1mQsV-JFAZ4HatdQyu-ZWaREkV4sF1avY": "ball_tuning_001/fold_6/model-034-0.891524.pth",
    "1G8ZMtLGSJO4JpZFd1SCOunhEGGGJcnyF": "ball_tuning_001/config.json",
    "1KWza3YseXeGLSaJWh0scU6rbOm417cfJ": "ball_tuning_001/source.py",
    # sampling_weights_001 — base ball model (2-class)
    "1drTggJ2UIpKpN4H4EU1sMnLOneOtts2r": "sampling_weights_001/fold_0/model-029-0.834085.pth",
    "1HSPUY_q6RBf1ssMb7sVYU8fXoUV6Ph-2": "sampling_weights_001/fold_1/model-029-0.871587.pth",
    "1RNEQQm_BvtCQJ3qSM0j9RvMrP5P0W_33": "sampling_weights_001/fold_2/model-029-0.784043.pth",
    "1a10K7PE0JqQguxPqRrA-23c9wkQEgmnK": "sampling_weights_001/fold_3/model-029-0.863879.pth",
    "1F9aQ5P0sq3KEPn6IFwUiijfcDmMbsSI-": "sampling_weights_001/fold_4/model-029-0.856508.pth",
    "1KiOEyTWf16Byogr-N2QvFB2UJ1ATiw4d": "sampling_weights_001/fold_5/model-029-0.883279.pth",
    "1KfT_MOaiNnAjUhLi7rpno9ap43ynIp8K": "sampling_weights_001/fold_6/model-029-0.883683.pth",
    "16Y-8jy7hUkGhsjW1RLR6ihitu9trPgi9": "sampling_weights_001/config.json",
    "1-0Qvb0O-ant4F1rOjAIXc9L9jVljzv_u": "sampling_weights_001/source.py",
    # ball_finetune_long_004 — longer fine-tune variant (2-class)
    "1uJEYltwSjOJmvfDY5SzZh7RL5j3-kyM_": "ball_finetune_long_004/fold_0/model-006-0.864002.pth",
    "1u6TdSDayZzYatsd2LJWSq_YBet4h1HIT": "ball_finetune_long_004/fold_1/model-006-0.862339.pth",
    "1gjAQSHhLFzvitz61ABxnn-_zCW8xrnpx": "ball_finetune_long_004/fold_2/model-006-0.758246.pth",
    "1d7uybKF1_3bFPE5llvuExtWk0oV-0bZy": "ball_finetune_long_004/fold_3/model-006-0.887217.pth",
    "1ahhX3sEE6DcuNlIJKOZcFTu-9gi3VxZ1": "ball_finetune_long_004/fold_4/model-006-0.869269.pth",
    "1raYTlfD1vGf7OVstK1K1m01aazcC3jKC": "ball_finetune_long_004/fold_5/model-006-0.901643.pth",
    "1vywJ-1ZFXexD9ReN5SSltF2OLFHGV-og": "ball_finetune_long_004/fold_6/model-006-0.897602.pth",
    "15y3_Od00O9joLPYAFDTRsD006pdS44C1": "ball_finetune_long_004/config.json",
    "15mP1W31_QRiXcZ_eCoa91NK1MSLGswca": "ball_finetune_long_004/source.py",
}


def main() -> int:
    import gdown

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="Restrict to one bundle name (path prefix)")
    ap.add_argument("--weights-only", action="store_true",
                    help="Skip config.json / source.py")
    args = ap.parse_args()

    ok = skip = fail = 0
    for fid, rel in FILES.items():
        if args.only and not rel.startswith(args.only):
            continue
        if args.weights_only and not rel.endswith(".pth"):
            continue
        out = DEST / rel
        if out.exists() and out.stat().st_size > 0:
            skip += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = gdown.download(id=fid, output=str(out), quiet=True)
            if r:
                ok += 1
                print(f"OK   {rel}  ({out.stat().st_size} B)")
            else:
                fail += 1
                print(f"FAIL {rel}  (quota? retry after ~24h)")
        except Exception as e:  # noqa: BLE001
            fail += 1
            print(f"ERR  {rel}: {e}")
    print(f"\n{ok} downloaded, {skip} already present, {fail} failed")
    if fail:
        print("Failures are usually Google Drive's daily quota — re-run tomorrow.")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
