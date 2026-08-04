#!/usr/bin/env python3
"""Rewrite the video/image URLs in a Label Studio tasks JSON.

The clips are expensive to render (~15 min for a 12-window match) and the URL in
the tasks file is the only thing that has to change when Label Studio can't
reach them. So this retargets an existing export in place of a re-dump.

A blank player with *"There has been an error rendering your video, please check
the format is supported"* is almost never the format — it is Label Studio
failing to fetch the file and the ``<video>`` element reporting the resulting
404/403 the only way it can. Check the codec first (``ffprobe``), and when it
comes back H.264 / yuv420p, the URL is what's wrong.

Two schemes:

``--doc-root DIR``
    Keep the built-in ``/data/local-files/?d=`` route, but recompute the ``?d=``
    path relative to the ``LOCAL_FILES_DOCUMENT_ROOT`` you actually intend to
    run with. The path is relative to the document root, not to the tasks file,
    so a dump nested one level deeper than the last one silently changes it.

``--base-url URL``
    Serve the clips over plain HTTP instead and sidestep local-files entirely —
    no document root, no serving flag, no storage connection to register. Pair
    with ``scripts/serve_clips.py``, which unlike ``python -m http.server`` sends
    the CORS and Range headers the player needs.

    soccer-video-analysis $ python scripts/retarget_label_studio_tasks.py \\
        runs/saints-u14g-full/tracklets/label_studio_tasks.json \\
        --base-url http://localhost:8000

The root the *existing* urls resolve against is found by walking up from the
tasks file until the paths land on real files, so it never has to be declared —
which matters because getting it wrong is the bug this script exists to repair.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

LOCAL_FILES_PREFIX = "/data/local-files/?d="
# The keys `enroll` puts a servable path under. Video for tracklet windows,
# image for the whole-frame route.
URL_KEYS = ("video", "image")


def current_relpath(url: str) -> str:
    """The servable path a task URL points at, whichever scheme wrote it.

    Both schemes encode the same thing — a path relative to whatever root is
    being served — so retargeting only needs that suffix back out.
    """
    if url.startswith(LOCAL_FILES_PREFIX):
        return unquote(url[len(LOCAL_FILES_PREFIX):])
    return unquote(urlsplit(url).path).lstrip("/")


def detect_current_root(tasks: list, tasks_path: Path) -> Path:
    """Find the root the existing urls are relative to, by walking up from the file.

    The tasks file sits inside the dump it describes, so the root is always an
    ancestor of it — ``runs/<match>/tracklets/label_studio_tasks.json`` holding
    ``?d=tracklets/clips/...`` resolves two levels up. Probing beats trusting a
    flag: a wrong root is exactly the failure being repaired, and it would be
    reintroduced silently by pointing every url at a file that doesn't exist.
    """
    rels = [current_relpath(t["data"][k]) for t in tasks
            for k in URL_KEYS if k in t.get("data", {})]
    if not rels:
        raise SystemExit(f"No {'/'.join(URL_KEYS)} urls in {tasks_path}")
    for root in [tasks_path.resolve().parent, *tasks_path.resolve().parents]:
        if all((root / rel).exists() for rel in rels):
            return root
    raise SystemExit(
        f"Could not locate the clips for {tasks_path}. The urls point at "
        f"'{rels[0]}', which exists under no ancestor of the tasks file — the "
        f"clips folder has probably been renamed or moved away from it.")


def retarget(url: str, *, clips_root: Path, new_root: Path,
             base_url: str | None) -> str:
    """Point ``url`` at the same file, addressed from ``new_root``."""
    abs_path = (clips_root / current_relpath(url)).resolve()
    try:
        rel = abs_path.relative_to(new_root.resolve()).as_posix()
    except ValueError:
        raise SystemExit(
            f"{abs_path} is not under {new_root.resolve()}; Label Studio can only "
            f"serve files beneath the root it is given") from None
    if base_url:
        return f"{base_url.rstrip('/')}/{rel}"
    return f"{LOCAL_FILES_PREFIX}{rel}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tasks", type=Path, help="label_studio_tasks.json to rewrite")
    dest = ap.add_mutually_exclusive_group(required=True)
    dest.add_argument("--doc-root", type=Path,
                      help="new LOCAL_FILES_DOCUMENT_ROOT to address files from")
    dest.add_argument("--base-url", metavar="URL",
                      help="serve over HTTP instead, e.g. http://localhost:8000")
    ap.add_argument("--http-root", type=Path,
                    help="folder you will serve for --base-url "
                         "(default: the tasks file's own folder)")
    ap.add_argument("--out", type=Path,
                    help="write here instead of overwriting the input")
    args = ap.parse_args()

    tasks = json.loads(args.tasks.read_text())
    current_root = detect_current_root(tasks, args.tasks)
    if args.base_url:
        new_root = (args.http_root or args.tasks.parent).resolve()
    else:
        new_root = args.doc_root.resolve()
    print(f"Current urls resolve against {current_root}")

    n = 0
    for task in tasks:
        data = task.get("data", {})
        for key in URL_KEYS:
            if key in data:
                data[key] = retarget(data[key], clips_root=current_root,
                                     new_root=new_root, base_url=args.base_url)
                n += 1

    out = args.out or args.tasks
    out.write_text(json.dumps(tasks, indent=2))
    print(f"Retargeted {n} url(s) across {len(tasks)} task(s) → {out}")
    print(f"  now: {tasks[0]['data'][next(k for k in URL_KEYS if k in tasks[0]['data'])]}")
    if args.base_url:
        port = args.base_url.rsplit(":", 1)[-1].rstrip("/") or "8000"
        print("\nServe the clips, then import this file:")
        print(f"  python scripts/serve_clips.py {new_root} --port {port}")
    else:
        print("\nStart Label Studio against the matching root, then import this file:")
        print("  export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true")
        print(f"  export LOCAL_FILES_DOCUMENT_ROOT={new_root.resolve()}")
        print("  label-studio start")
        print("  # Settings → Cloud Storage → Add Source Storage → Local files,")
        print("  # path = the same root. Recent Label Studio refuses to serve a")
        print("  # file that no storage connection covers, even under the root.")


if __name__ == "__main__":
    main()
