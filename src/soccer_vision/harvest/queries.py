"""Default search queries for harvesting youth-soccer footage.

Diversity is the whole point: the list spans languages, age groups, and terms
so the resulting clip set covers many cameras, countries, and kit colours
rather than over-representing one league or uploader. Callers can override the
list with ``--queries``/``--queries-file``.

Queries are searched in order and the run stops at the target, so the list is
ordered by priority: elevated auto-tracking cameras first (Veo / XbotGo /
XbotFalcon / Trace produce the high-vantage tactical view we want for analysis,
and their user base is overwhelmingly youth/amateur), then general youth-soccer
terms. English queries say "soccer" (not "football") to avoid pulling American
football; the non-English terms carry the linguistic diversity.
"""

from __future__ import annotations

# Elevated auto-camera brands. These give the high vantage point for analysis
# and skew youth/non-professional. Searched first so they fill the set before
# the generic queries.
PRIORITY_QUERIES: list[str] = [
    "veo soccer full match",
    "veo youth soccer full match",
    "xbotgo soccer full match",
    "xbotgo youth soccer",
    "xbotgo falcon soccer",       # XbotFalcon = XbotGo's Falcon model
    "xbotgo chameleon soccer",    # XbotGo's other auto-tracking model
    "trace soccer full match",
    "trace up soccer full match",
    "pixellot soccer youth",      # another elevated auto-camera brand
    "youth soccer full match tactical camera",
    "u12 soccer full match veo",
]

# General youth / grassroots soccer, multi-lingual for camera/country/kit spread.
GENERAL_QUERIES: list[str] = [
    # English (always "soccer" to keep American football out)
    "u10 soccer full match",
    "u11 youth soccer full game",
    "u12 soccer full match",
    "u13 soccer full game",
    "u14 girls soccer full match",
    "u15 academy soccer full match",
    "youth soccer full match",
    "junior soccer full game",
    "grassroots soccer full match",
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

DEFAULT_QUERIES: list[str] = PRIORITY_QUERIES + GENERAL_QUERIES
