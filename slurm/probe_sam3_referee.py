"""Can SAM3 separate referees from players — and can it do both in ONE pass?

Decides the ref-filter architecture:
  (A) multi-prompt: add_text_prompt(session, ["soccer player","referee"]) in one
      pass, IF the output attributes objects to a prompt -> free.
  (B) two passes, but the ref pass only on a SAMPLE of frames, matching ref
      masks to persistent player ids by IoU -> minutes, not a 2nd 3h run.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

VIDEO = "/orange/ewhite/b.weinstein/soccer-video-analysis/data/SaintsU11_OVF_Jul192026.MP4"
START, N = 15000, 12
MODEL = "facebook/sam3"


def clip():
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, START)
    fr = []
    for _ in range(N):
        ok, f = cap.read()
        if not ok:
            break
        fr.append(np.ascontiguousarray(f[:, :, ::-1]))
    cap.release()
    return fr


def run(processor, model, frames, prompt, device, dtype):
    session = processor.init_video_session(video=frames, inference_device=device, dtype=dtype)
    processor.add_text_prompt(session, prompt)
    per_frame = []
    with torch.inference_mode():
        for i in range(len(frames)):
            out = model(inference_session=session, frame_idx=i)
            per_frame.append([int(o) for o in out.object_ids])
    return per_frame, out


def main():
    device, dtype = "cuda", torch.bfloat16
    frames = clip()
    processor = Sam3VideoProcessor.from_pretrained(MODEL)
    model = Sam3VideoModel.from_pretrained(MODEL, dtype=dtype).to(device).eval()

    print("=== A) single prompt 'referee' ===")
    ref_pf, ref_out = run(processor, model, frames, "referee", device, dtype)
    print("   referees/frame:", [len(x) for x in ref_pf])
    print("   distinct ref ids:", len({i for f in ref_pf for i in f}))

    print("\n=== B) single prompt 'soccer player' (reference) ===")
    ply_pf, _ = run(processor, model, frames, "soccer player", device, dtype)
    print("   players/frame:", [len(x) for x in ply_pf])

    print("\n=== C) MULTI-prompt ['soccer player','referee'] in one pass ===")
    try:
        multi_pf, multi_out = run(processor, model, frames,
                                  ["soccer player", "referee"], device, dtype)
        print("   objects/frame:", [len(x) for x in multi_pf])
        print("   distinct ids:", len({i for f in multi_pf for i in f}))
        # Does the output attribute an object to which prompt matched?
        attrs = [a for a in dir(multi_out) if not a.startswith("_")]
        print("   output fields:", attrs)
        for cand in ("obj_id_to_prompt", "obj_id_to_label", "obj_id_to_text",
                     "prompt_ids", "obj_id_to_concept"):
            if hasattr(multi_out, cand):
                print(f"   >>> ATTRIBUTION FIELD FOUND: {cand} = "
                      f"{getattr(multi_out, cand)}")
        print("   VERDICT: multi-prompt runs; attribution field present above (if any)")
    except Exception as e:
        print(f"   multi-prompt FAILED: {type(e).__name__}: {e}")
        print("   -> use approach (B): sampled 2nd pass + IoU match")


if __name__ == "__main__":
    main()
