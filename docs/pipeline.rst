Pipeline
========

The canonical pipeline processes every match video through 9 steps:

1. **Load raw video** — read metadata and validate input
2. **Virtual broadcast** — generate follow-cam 16:9 proxy from wide-angle footage
3. **Ball detection** — RF-DETR on broadcast proxy
4. **Player tracking** — ByteTrack multi-object tracking
5. **Field registration** — Hough-line homography (sn-calibration fallback)
6. **Event detection** — set-piece heuristics + spotting model adapters
7. **Team metrics** — distance, possession, shots, heatmaps
8. **Database logging** — SQLite + OSL JSON export
9. **Clip creation** — ffmpeg extraction + highlight reels
