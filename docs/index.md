# soccer-vision

Turn a single-camera youth match into the clips you actually want to watch.
Point it at Veo, overhead, or any fixed wide-angle footage. Python 3.12+,
runs on a CPU, no cloud, no subscription.

![A shot on goal, with every player, the keeper, the referee and the ball tracked frame to frame](images/perception_clip.gif)

Two questions drive everything:

| Question | Page |
|---|---|
| *"Every touch by number 6."* | [Individual highlights](individual-highlights.md) |
| *"Everything the black team did on the ball."* | [Team highlights](team-highlights.md) |

```bash
pip install -e .
soccer-vision process match.mp4 --match-id match_001
soccer-vision reel --run runs/match_001 --number 6 --halo --out number6.mp4
```

:::{note}
Selecting by **player** or **team** works today. Selecting by **action label**
(`--events pass`) needs an action detector, which is the piece still being
built — see [Training new detectors](training-detectors.md).
:::

```{toctree}
:maxdepth: 1
:caption: Start here

about
installation
```

```{toctree}
:maxdepth: 1
:caption: Making clips

individual-highlights
team-highlights
profiles
```

```{toctree}
:maxdepth: 1
:caption: Improving it

training-detectors
contributing
```
