"""Default search queries for harvesting youth-soccer footage.

Diversity is the whole point: the list spans languages, age groups, and terms
so the resulting clip set covers many cameras, countries, and kit colours
rather than over-representing one league or uploader. Callers can override the
list with ``--queries``/``--queries-file``.
"""

from __future__ import annotations

# Kept deliberately broad and multi-lingual. "full match"/"full game" phrasing
# biases toward continuous fixed-camera match footage (what we can annotate)
# over highlight montages and talking-head content.
DEFAULT_QUERIES: list[str] = [
    # English
    "u10 soccer full match",
    "u11 youth soccer full game",
    "u12 football full match",
    "u13 soccer full game",
    "u14 girls soccer full match",
    "u15 academy football full match",
    "youth soccer full match",
    "junior football full game",
    "grassroots football full match",
    "high school soccer full game",
    # Spanish
    "sub 12 futbol partido completo",
    "sub 13 futbol infantil partido completo",
    "futbol base partido completo",
    "categoria alevin partido completo",
    # Portuguese
    "futebol sub 13 jogo completo",
    "futebol de base jogo completo",
    # German
    "e jugend fussball ganzes spiel",
    "d jugend fussball spiel",
    "jugendfussball ganzes spiel",
    # French
    "football u13 match complet",
    "football des jeunes match complet",
    # Italian / Dutch / Nordic
    "calcio giovanile partita completa",
    "jeugdvoetbal hele wedstrijd",
    "ungdomsfotball hel kamp",
]
