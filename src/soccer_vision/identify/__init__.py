"""Player identity: read jersey numbers per track and resolve names.

Powers the *individual-player* query pathway ("show me number six's passes").
Team-level queries use jersey *colour* (:mod:`soccer_vision.tracking.teams`) and
need none of this.

The flow is: a dedicated jersey-number recognizer reads the digits off each
player crop frame by frame (:mod:`.jersey_ocr`), the noisy per-frame reads are
collapsed to one number per track by confidence-weighted voting (:mod:`.vote`),
and a name/number query is resolved to the set of track lanes carrying that
jersey (:mod:`.resolve`).
"""
