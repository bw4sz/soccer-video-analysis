"""soccer-vision annotate: set up Label Studio review + export fine-tune data.

Two modes:

* **build** (default) — turn a processed run into a Label Studio project:
  writes ``labeling_config.xml`` and ``label_studio_tasks.json`` (tasks
  pre-filled with the pipeline's — and, if present, SoccerChat's — predictions).
  With ``--push`` it creates the project on a running Label Studio server via the
  SDK and imports the tasks directly.

* **export** (``--export EXPORT.json``) — convert a Label Studio annotation
  export into ms-swift fine-tune records (``--finetune-out``), using the
  annotator's corrected labels. This is the youth-footage training set.
"""

from __future__ import annotations

import json
from pathlib import Path


def run_annotate(args):
    from soccer_vision.annotate import label_studio as ls

    if args.export:
        _run_export(args, ls)
    else:
        _run_build(args, ls)


def _run_export(args, ls):
    clips_root = args.clips_root
    if not clips_root and args.run:
        clips_root = str(Path(args.run) / "clips")

    records = ls.export_finetune(args.export, clips_root=clips_root)
    out = Path(args.finetune_out or "soccerchat_finetune.jsonl")
    with open(out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {len(records)} fine-tune record(s) → {out}")
    if not records:
        print("  (no annotated tasks found in the export — annotate some clips first)")


def _run_build(args, ls):
    if not args.run:
        print("annotate build needs --run <run-dir>.")
        return
    run_dir = Path(args.run)
    if not (run_dir / "annotations.json").exists():
        print(f"No annotations.json in {run_dir}. Run 'soccer-vision process' first.")
        return

    paths = ls.write_project_files(run_dir, serve_root=args.serve_root, out_dir=args.out)
    n = paths["_n_tasks"]
    print(f"Label Studio project files for {n} clip(s):")
    print(f"  config: {paths['config']}")
    print(f"  tasks:  {paths['tasks']}")

    if args.push:
        _push(args, ls, paths, n)
    else:
        serve_root = args.serve_root or str(run_dir.parent)
        print("\nTo review (offline import):")
        print("  export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true")
        print(f"  export LOCAL_FILES_DOCUMENT_ROOT={Path(serve_root).resolve()}")
        print("  label-studio start")
        print("  # then in the UI: create project → paste config → import tasks.json")
        print("See label_studio/README.md for the full walkthrough.")


def _push(args, ls, paths, n):
    if not args.ls_url or not args.ls_key:
        print("\n--push needs --ls-url and --ls-key (Label Studio URL + API token).")
        return
    try:
        from label_studio_sdk import Client
    except Exception as exc:
        print(f"\nlabel-studio-sdk not installed ({exc}).")
        print("  uv sync --extra annotate   (or: pip install label-studio-sdk)")
        return

    tasks = json.loads(Path(paths["tasks"]).read_text())
    config = Path(paths["config"]).read_text()
    client = Client(url=args.ls_url, api_key=args.ls_key)
    title = args.title or f"soccer-vision {Path(args.run).name}"
    project = client.start_project(title=title, label_config=config)
    project.import_tasks(tasks)
    print(f"\nPushed project '{title}' with {n} task(s) → {args.ls_url}")
