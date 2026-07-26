"""Why does the SAM3 session OOM? Measure what actually grows, and whether
remove_object() reduces it. Two blind fixes failed; instrument instead.

Q1: does out.object_ids accumulate (all known objects) or reflect only
    currently-visible ones?
Q2: how many objects have a NON-EMPTY mask per frame (truly visible)?
Q3: does session.get_obj_num() grow, and does remove_object() shrink it
    and free GPU memory?
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

VIDEO = "/orange/ewhite/b.weinstein/soccer-video-analysis/data/SaintsU11_OVF_Jul192026.MP4"
START, N_FRAMES, STRIDE = 15000, 240, 6   # mirrors pipeline's 5fps sampling
MODEL = "facebook/sam3"


def main():
    device, dtype = "cuda", torch.bfloat16
    processor = Sam3VideoProcessor.from_pretrained(MODEL)
    model = Sam3VideoModel.from_pretrained(MODEL, dtype=dtype).to(device).eval()
    session = processor.init_video_session(inference_device=device, dtype=dtype)
    processor.add_text_prompt(session, "soccer player")

    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, START)

    last_seen: dict[int, int] = {}
    print(f"{'frm':>5}{'ids':>6}{'visible':>9}{'objnum':>8}{'gpuGB':>8}{'pruned':>8}")
    i = 0
    while i < N_FRAMES:
        for _ in range(STRIDE):
            ok, bgr = cap.read()
        if not ok:
            break
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        inputs = processor(images=rgb, return_tensors="pt")
        pv = inputs["pixel_values"].to(device, dtype=dtype)
        with torch.inference_mode():
            out = model(inference_session=session, frame=pv[0])
        i += 1

        ids = [int(o) for o in out.object_ids]
        # Q2: how many are actually visible (non-empty mask)?
        visible = []
        for oid in ids:
            m = out.obj_id_to_mask[oid]
            if isinstance(m, torch.Tensor):
                nz = bool((m > 0).any().item())
            else:
                nz = bool(np.any(np.asarray(m) > 0))
            if nz:
                visible.append(oid)
        for oid in visible:
            last_seen[oid] = i

        pruned = 0
        if i % 40 == 0:
            # Q3: prune objects not VISIBLE recently, then re-measure
            stale = [o for o, t in last_seen.items() if i - t > 20]
            unseen = [o for o in ids if o not in last_seen]
            for oid in stale + unseen:
                try:
                    session.remove_object(oid)
                    pruned += 1
                except Exception:
                    pass
                last_seen.pop(oid, None)

        if i % 10 == 0 or pruned:
            print(f"{i:>5}{len(ids):>6}{len(visible):>9}"
                  f"{session.get_obj_num():>8}"
                  f"{torch.cuda.memory_allocated()/1e9:>8.2f}{pruned:>8}")

    cap.release()
    print("\nCONCLUSION INPUTS:")
    print(f"  final ids in last frame : {len(ids)}")
    print(f"  final visible           : {len(visible)}")
    print(f"  final session obj_num   : {session.get_obj_num()}")
    print(f"  peak GPU alloc (GB)     : {torch.cuda.max_memory_allocated()/1e9:.2f}")


if __name__ == "__main__":
    main()
