"""Fetch the FOOTPASS / SN-PCBAS-2026 dataset from HuggingFace into the layout
the FOOTPASS TAAD dataloader expects.

The data lives at https://huggingface.co/datasets/SoccerNet/SN-PCBAS-2026 as a
**gated** dataset (gated: auto). To use this script you must, once:
  1. Create a HuggingFace account.
  2. Accept the terms on the dataset page (auto-granted).
  3. `hf auth login`  (or set $HF_TOKEN) so downloads are authenticated.

Caveat: the SoccerNet NDA note says a password may be required to *extract* the
zips. If extraction fails with a password error, pass --zip-password (defaults to
the repo's .soccernet_token if present).

The FOOTPASS TAAD dataloader (`vendor/FOOTPASS/utils/TAAD_Dataset.py`) reads,
relative to --dest (its `data_root`):
    data/train_tactical_data.h5, data/val_tactical_data.h5
    data/TAAD_sample_list.json          (ships in the repo; copied here)
    videos/game_<idx>.mp4

Usage:
    hf auth login
    python scripts/footpass_fetch_data.py                       # train+val, 352x640
    python scripts/footpass_fetch_data.py --resolution fullHD
    python scripts/footpass_fetch_data.py --splits train val challenge
    python scripts/footpass_fetch_data.py --list-only           # just show what it'd fetch
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FOOTPASS = REPO / "vendor" / "FOOTPASS"
HF_REPO = "SoccerNet/SN-PCBAS-2026"
DEFAULT_DEST = Path("/blue/ewhite/b.weinstein/soccer-vision-data/footpass")

# HF filenames per split. fullHD TRAIN is split into 5 parts.
VIDEO_FILES = {
    ("352x640", "TRAIN"): ["videos_352x640_TRAIN.zip"],
    ("352x640", "VAL"): ["videos_352x640_VAL.zip"],
    ("352x640", "CHALLENGE"): ["videos_352x640_CHALLENGE.zip"],
    ("fullHD", "TRAIN"): [f"videos_fullHD_TRAIN_{i:02d}.zip" for i in range(1, 6)],
    ("fullHD", "VAL"): ["videos_fullHD_VAL.zip"],
    ("fullHD", "CHALLENGE"): ["videos_fullHD_CHALLENGE.zip"],
}


def resolve_zip_password(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    token = REPO / ".soccernet_token"
    if token.exists():
        return token.read_text().strip()
    return None


def _is_aes(zip_path: Path) -> bool:
    """WinZip AES entries report compress_type 99 — stdlib zipfile can't read them."""
    with zipfile.ZipFile(zip_path) as zf:
        return any(i.compress_type == 99 for i in zf.infolist())


def extract(zip_path: Path, out_dir: Path, password: str | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pwd = password.encode() if password else None
    # The FOOTPASS *video* zips are WinZip-AES encrypted (method 99); stdlib
    # zipfile (and system `unzip`) can't decrypt them. Use pyzipper for those.
    if _is_aes(zip_path):
        try:
            import pyzipper
        except ImportError:
            sys.exit(f"{zip_path.name} is AES-encrypted — `uv pip install pyzipper`.")
        if pwd is None:
            sys.exit(f"{zip_path.name} is AES-encrypted; provide the SoccerNet "
                     "password via --zip-password or the gitignored .soccernet_token.")
        with pyzipper.AESZipFile(zip_path) as zf:
            zf.extractall(out_dir, pwd=pwd)
    else:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir, pwd=pwd)
    print(f"  extracted {zip_path.name} -> {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help=f"data_root to populate (default {DEFAULT_DEST})")
    ap.add_argument("--splits", nargs="+", default=["train", "val"],
                    choices=["train", "val", "challenge"])
    ap.add_argument("--resolution", default="352x640", choices=["352x640", "fullHD"])
    ap.add_argument("--zip-password", default=None,
                    help="Password to extract zips (default: repo .soccernet_token)")
    ap.add_argument("--no-videos", action="store_true", help="Only tactical data + annotations")
    ap.add_argument("--list-only", action="store_true", help="Print the plan and exit")
    args = ap.parse_args()

    splits = [s.upper() for s in args.splits]
    wanted = ["tactical_data_format.txt"]
    for s in splits:
        wanted.append(f"tactical_data_{s}.zip")
    if not args.no_videos:
        for s in splits:
            wanted += VIDEO_FILES[(args.resolution, s)]

    print(f"HF repo : {HF_REPO}")
    print(f"dest    : {args.dest}")
    print(f"splits  : {splits}  |  resolution: {args.resolution}")
    print("files   :")
    for f in wanted:
        print(f"  - {f}")
    if args.list_only:
        return 0

    try:
        from huggingface_hub import get_token, hf_hub_download
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError
    except ImportError:
        sys.exit("pip/uv install huggingface_hub")

    if not get_token():
        sys.exit(
            "No HuggingFace token found. Authenticate first (the `.venv/bin/hf` on "
            "PATH is broken — use the blue env or $HF_TOKEN):\n"
            "  /blue/ewhite/b.weinstein/envs/soccer-vision/bin/hf auth login\n"
            "  # or:  export HF_TOKEN=hf_xxx\n"
            f"and accept terms at https://huggingface.co/datasets/{HF_REPO}"
        )

    data_dir = args.dest / "data"
    videos_dir = args.dest / "videos"
    data_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Copy the repo-shipped annotations so `dest` is a complete data_root.
    for rel in ["data/TAAD_sample_list.json"]:
        src = FOOTPASS / rel
        if src.exists():
            shutil.copy2(src, data_dir / src.name)
            print(f"copied {rel} -> {data_dir/src.name}")
    if (FOOTPASS / "playbyplay_GT").exists():
        shutil.copytree(FOOTPASS / "playbyplay_GT", args.dest / "playbyplay_GT",
                        dirs_exist_ok=True)

    zip_pw = resolve_zip_password(args.zip_password)
    cache = args.dest / "_hf_cache"
    for fname in wanted:
        print(f"\n[{fname}] downloading...")
        try:
            p = hf_hub_download(repo_id=HF_REPO, filename=fname, repo_type="dataset",
                                local_dir=str(cache))
        except GatedRepoError:
            sys.exit(f"GATED: accept terms at https://huggingface.co/datasets/{HF_REPO} "
                     "and run `hf auth login`.")
        except HfHubHTTPError as e:
            if "401" in str(e) or "403" in str(e):
                sys.exit("AUTH: not logged in / terms not accepted. "
                         f"Accept https://huggingface.co/datasets/{HF_REPO} then `hf auth login`.")
            raise
        p = Path(p)
        if fname.endswith(".zip"):
            out = data_dir if fname.startswith("tactical") else videos_dir
            extract(p, out, zip_pw)
        else:
            shutil.copy2(p, data_dir / Path(fname).name)

    print(f"\nDone. Train with: --data_root {args.dest} "
          f"(from vendor/FOOTPASS: `python train_TAAD_Baseline.py --data_root {args.dest}`)")
    print("NOTE: dataloader expects data/train_tactical_data.h5 & data/val_tactical_data.h5 — "
          "verify the extracted names match; rename if the zip uses a different filename.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
