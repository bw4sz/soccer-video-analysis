Pipeline
========

The canonical pipeline processes every match video through 9 steps:

1. **Load raw video** — read metadata and validate input
2. **Virtual broadcast** — generate follow-cam 16:9 proxy from wide-angle footage
3. **Ball detection** — RF-DETR on broadcast proxy
4. **Player tracking** — ByteTrack multi-object tracking
5. **Field registration** — Hough-line homography (sn-calibration fallback)
6. **Event detection** — pluggable event *sources* (set-piece heuristics today;
   the T-DEED team tackle model when trained), then **player/team association**
7. **Team metrics** — distance, possession, shots, heatmaps
8. **Database logging** — SQLite + OSL JSON export
9. **Clip creation** — ffmpeg extraction + highlight reels

Event sources, team/player association, and clips
--------------------------------------------------

Detection is decoupled from association and clip selection so new models drop in
without touching downstream code:

- **Event sources** (:mod:`soccer_vision.events.sources`) implement a common
  ``EventSource`` interface (``is_available`` / ``detect``). ``SetPieceSource``
  wraps the ball-position heuristics; ``TackleSource`` is the interface for the
  T-DEED team tackle model (``training/sn_spotting/train_teamspotting.py``,
  label ``PLAYER SUCCESSFUL TACKLE`` → ``tackle``) — it activates once a
  checkpoint is registered, with no changes to association or clips. Events are
  event-type-agnostic (``tackle``, ``goal``, ``goal_kick``, ...).
- **Team assignment** (:mod:`soccer_vision.tracking.teams`): v1 clusters tracked
  players into two teams by jersey colour and names each cluster (blue / white /
  ...), so events can be filtered by ``--team blue``. Individual-player identity
  comes from ``soccer-vision identify`` (:mod:`soccer_vision.identify`), which
  reads jersey numbers per track (dedicated recognizer → confidence-weighted
  vote) so events filter by ``--player`` / ``--number``; SAM3 masklet identity is
  a later phase behind the same seam.
- **Association** (:mod:`soccer_vision.events.associate`): each event is tagged
  with the nearest player's ``track_id`` and their ``team``.
- **On-ball spans** (:mod:`soccer_vision.events.on_ball`): the frames where a
  given player is the ball's nearest player, computed from ``ball_track.json``
  and ``tracks.json`` in pixel space. Independent of both the event detector and
  the field homography, and the fallback behind ``--player`` (below).

Cut clips for a whole team or a single player (track), composable with the event
label::

    soccer-vision reel --run runs/<id> --team blue --out blue_team.mp4
    soccer-vision reel --run runs/<id> --event tackle --team blue
    soccer-vision extract --run runs/<id> --track 7 --events tackle goal_kick

.. _on-ball-fallback:

On-ball fallback — what ``--player`` does when no event matched
---------------------------------------------------------------

Today's rules engine only fires on set pieces, so asking for one player's clips
almost always matches nothing in the event stream: a youth match yields a
handful of throw-ins and goal kicks, and any one player is the nearest player
for only a few of them. Coming back empty is technically correct and useless.

So when a **player selection** (``--player`` / ``--number`` / ``--track``)
matches no detected events, ``extract`` and ``reel`` fall back to
:func:`soccer_vision.events.on_ball.select_on_ball_spans` — every span where
that player was the ball's nearest player, within ``--on-ball-dist`` pixels.
These are emitted as ordinary ``on_ball`` events, so ``--team``, ``--halo``, and
clip naming all work unchanged::

    # No pass detector yet — this cuts Simon's touches
    soccer-vision reel --run runs/<id> --player Simon --profile team.yaml --halo

    # Force it even though an event label was given
    soccer-vision extract --run runs/<id> --player Simon --events pass --on-ball

    # Opt out — report nothing rather than substituting proximity
    soccer-vision extract --run runs/<id> --player Simon --no-on-ball

Two deliberate limits:

- The fallback **does not fire when an explicit event label was requested**.
  ``--events pass`` returning ball-proximity touches would answer a different
  question than the one asked; pass ``--on-ball`` to override.
- It **needs a player selection**. ``--team blue`` alone is a team query with no
  single lane to anchor spans on, so it is left to the event stream.

A player fragments across several ByteTrack lanes, and a single continuous touch
can cross a handoff. Spans are *not* split there — that would cut one action into
two clips — so each span carries every lane it covers in ``track_ids`` and the
halo follows all of them. ``track_id`` is the lane that got closest to the ball.

Requires ``ball_track.json`` and ``tracks.json``; runs predating those files
report what is missing rather than silently reading as "this player did
nothing".
