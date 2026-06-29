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
