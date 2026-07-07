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

Cut clips for a whole team or a single player (track), composable with the event
label::

    soccer-vision reel --run runs/<id> --team blue --out blue_team.mp4
    soccer-vision reel --run runs/<id> --event tackle --team blue
    soccer-vision extract --run runs/<id> --track 7 --events tackle goal_kick
