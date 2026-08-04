#!/usr/bin/env python3
"""Serve a folder of clips to Label Studio over plain HTTP.

An alternative to Label Studio's ``/data/local-files/`` route, which needs an
environment variable set before the server starts, a document root at exactly
the right depth, *and* a registered storage connection — three things that fail
silently and identically, as a video player reporting a format error.

``python -m http.server`` is the obvious substitute and doesn't work here, for
two reasons this fixes:

* **No CORS.** The clips come off :8000 while Label Studio runs on :8080, and
  its player reads frames through a canvas, so the fetch is subject to
  cross-origin rules that plain ``<video>`` playback would escape.
* **No Range support.** ``SimpleHTTPRequestHandler`` answers every request with
  the whole file and ignores ``Range``, so scrubbing a 20 s window either
  restarts it or hangs — and scrubbing is the point of labelling tracklets.

    $ python scripts/serve_clips.py runs/saints-u14g-full/tracklets
    Serving .../tracklets on http://localhost:8000  (12 mp4)

Leave it running in its own shell. Read-only: it never writes to the directory.
"""
from __future__ import annotations

import argparse
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class ClipHandler(SimpleHTTPRequestHandler):
    """Static handler with CORS and byte-range support."""

    # Scrubbing a clip fires a burst of small range requests, and under the
    # default HTTP/1.0 each one pays a fresh TCP connection. Safe to raise here
    # because every response below sends an accurate Content-Length.
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_OPTIONS(self):  # noqa: N802 — http.server's naming
        self.send_response(204)
        self.end_headers()

    def send_head(self):
        """Answer a ``Range`` request with 206 and the requested slice."""
        rng = self.headers.get("Range")
        if not rng:
            return super().send_head()

        m = RANGE_RE.fullmatch(rng.strip())
        path = self.translate_path(self.path)
        if not m or os.path.isdir(path):
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        size = os.fstat(f.fileno()).st_size
        first, last = m.group(1), m.group(2)
        if first:
            # "bytes=N-" and "bytes=N-M" — an explicit start.
            start = int(first)
            end = min(int(last), size - 1) if last else size - 1
        else:
            # "bytes=-N" is the *suffix*: the final N bytes, which is how a
            # player finds a moov atom that isn't at the front.
            start = max(0, size - int(last or 0))
            end = size - 1
        if start >= size or start > end:
            f.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None

        f.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        # Truncate the body to the slice: copyfile() would stream to EOF.
        return _Slice(f, end - start + 1)


class _Slice:
    """File wrapper that yields at most ``remaining`` bytes to ``copyfile``."""

    def __init__(self, fh, remaining: int):
        self._fh, self._remaining = fh, remaining

    def read(self, n: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        n = self._remaining if n < 0 else min(n, self._remaining)
        chunk = self._fh.read(n)
        self._remaining -= len(chunk)
        return chunk

    def close(self):
        self._fh.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", type=Path, help="folder to serve (the one the "
                                                 "task urls are relative to)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    root = args.directory.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")
    n_clips = len(list(root.rglob("*.mp4")))

    handler = partial(ClipHandler, directory=str(root))
    with ThreadingHTTPServer(("", args.port), handler) as httpd:
        print(f"Serving {root} on http://localhost:{args.port}  ({n_clips} mp4)")
        print("Ctrl-C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
