Project Profiles
================

A profile YAML file customizes soccer-vision for a specific team::

    team_name: "Saints U10 Azul"
    season: "2026-spring"
    roster:
      - name: Noah
        jersey: 7
        role: holding_mid

Use with::

    soccer-vision process match.mp4 --profile saints-u10.yaml

The ``roster`` maps each jersey number to a name. To select clips by name
(``--player Noah``) rather than number, run ``soccer-vision identify`` first — it
reads jersey numbers off the tracks into ``jerseys.json``; the profile then
resolves the name to that number::

    soccer-vision identify --run runs/<id> --profile saints-u10.yaml
    soccer-vision reel --run runs/<id> --player Noah --profile saints-u10.yaml
