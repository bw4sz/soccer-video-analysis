"""Where the labelled crops for a gallery come from.

Two ways to tell the gallery "this appearance is Simon":

**Annotate frames in Label Studio** — the recommended one. ``enroll
--dump-frames`` exports whole frames with every detected player pre-boxed; you
name the boxes that are your squad and delete the rest. Ground truth, and the
only option for a player OCR never reads (a keeper in a different kit, a number
that faces away all match). Identity is judged on the full frame while the model
still crops at the size it trains on, which is why this beat labelling the crops
themselves — a player is ~29 px tall on overhead footage and unrecognisable in
isolation at any zoom.

**Bootstrap from OCR** — reuse the jersey numbers `identify` already voted. Only
high-confidence tracks are taken, so the gallery is seeded from the reads OCR got
*right* and then generalises to the many crops it couldn't read at all. Costs no
annotation, but a confident-but-wrong vote enrols the wrong player, so the
confidence floor is deliberately high.

Parsing here ends at ``(frame, bbox, name)`` triples that the CLI embeds. No
model and no video, so it stays testable.
"""

from __future__ import annotations

import numpy as np

from soccer_vision.profiles.loader import get_player


def names_from_jerseys(
    jerseys_doc: dict,
    profile: dict | None = None,
    *,
    min_confidence: float = 0.8,
    min_obs: int = 5,
    exclude: set[int] | None = None,
) -> dict[int, str]:
    """``{track_id: player_name}`` for tracks OCR named confidently enough.

    A track qualifies only if its vote cleared ``min_confidence`` on at least
    ``min_obs`` legible reads. ``exclude`` drops jersey numbers you don't trust —
    on Veo footage PARSeq hallucinates a small set of digits on unreadable backs,
    and one such number enrolled as a player poisons every match afterwards.

    Names come from the profile roster; a jersey with no roster entry enrols as
    ``"#7"`` so an unrostered squad still gets per-player identity.
    """
    exclude = exclude or set()
    out: dict[int, str] = {}
    for tid, info in jerseys_doc.get("tracks", {}).items():
        jersey = info.get("jersey")
        if jersey is None or jersey in exclude:
            continue
        if info.get("confidence", 0.0) < min_confidence or info.get("n_obs", 0) < min_obs:
            continue
        player = get_player(profile, jersey) if profile else None
        out[int(tid)] = (player or {}).get("name") or f"#{jersey}"
    return out


def boxes_from_label_studio(export: list[dict]) -> list[tuple[int, np.ndarray, str]]:
    """Parse a Label Studio ``rectanglelabels`` export to ``(frame, bbox, name)``.

    Expects one task per sampled frame, carrying the frame number in
    ``data.frame`` (or a numeric filename stem), with each rectangle labelled by
    the player's name. Label Studio stores boxes as percentages of the image, so
    they're scaled back to pixels here. Set up the project with::

        <View><Image name="img" value="$image"/>
          <RectangleLabels name="player" toName="img">
            <Label value="Simon Weinstein"/> ...
          </RectangleLabels></View>
    """
    out: list[tuple[int, np.ndarray, str]] = []
    for task in export:
        frame = _task_frame(task)
        if frame is None:
            continue
        for ann in task.get("annotations", []) or []:
            for res in ann.get("result", []) or []:
                value = res.get("value", {})
                labels = value.get("rectanglelabels") or []
                if not labels:
                    continue
                w_px = float(res.get("original_width") or 0)
                h_px = float(res.get("original_height") or 0)
                if w_px <= 0 or h_px <= 0:
                    continue
                x = value["x"] / 100.0 * w_px
                y = value["y"] / 100.0 * h_px
                w = value["width"] / 100.0 * w_px
                h = value["height"] / 100.0 * h_px
                bbox = np.asarray([x, y, x + w, y + h], dtype=np.float32)
                out.append((frame, bbox, str(labels[0])))
    return out


def _task_frame(task: dict) -> int | None:
    """Frame number for a task: explicit ``data.frame``, else a numeric stem."""
    data = task.get("data", {}) or {}
    if data.get("frame") is not None:
        return int(data["frame"])
    for value in data.values():
        stem = str(value).rsplit("/", 1)[-1].rsplit(".", 1)[0]
        digits = "".join(c for c in stem if c.isdigit())
        if digits:
            return int(digits)
    return None
