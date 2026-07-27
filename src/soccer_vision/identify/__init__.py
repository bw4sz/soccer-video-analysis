"""Player identity: read jersey numbers per track and resolve names.

Powers the *individual-player* query pathway ("show me number six's passes").
Team-level queries use jersey *colour* (:mod:`soccer_vision.tracking.teams`) and
need none of this.

There are two routes to a name, and :mod:`soccer_vision.cli.identify` can run
either or both:

**Appearance re-id** — the preferred route for a team you see every week. Enrol
each player's look once (:mod:`.enroll`), embed player crops with the sportsreid
backbone (:mod:`.reid`), and name a track by nearest-neighbour lookup in that
gallery (:mod:`.gallery`). No jersey has to be readable, so it survives the backs
and blurs that defeat OCR.

**Jersey OCR** — the cold-start route, and the fallback for anyone not enrolled.
A dedicated recognizer reads the digits off each player crop frame by frame
(:mod:`.jersey_ocr`) and the noisy per-frame reads are collapsed to one number
per track by confidence-weighted voting (:mod:`.vote`).

Either way, a name/number query is resolved to the set of track lanes carrying
that identity (:mod:`.resolve`).
"""
