"""Claude API integration for event verification and natural language queries.

verify_events():  send contact sheets + candidate events → get verified/rejected lists
query_match():    natural language query over OSL JSON + stats + optional profile
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any


def _client():
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=key)


def _encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def verify_events(
    sheet_paths: list[Path],
    candidate_events: list[dict],
    *,
    profile: dict | None = None,
    model: str = "claude-sonnet-4-6",
) -> dict[str, Any]:
    """Send contact sheets + events to Claude for verification.

    Returns:
        {
          "verified": [{"frame": int, "label": str, "reason": str}, ...],
          "rejected": [{"frame": int, "label": str, "reason": str}, ...],
          "raw_response": str,
        }
    """
    client = _client()

    roster_text = ""
    if profile and profile.get("roster"):
        lines = [f"  - #{p['jersey']} {p['name']} ({p.get('role', '')})"
                 for p in profile["roster"]]
        roster_text = "Roster:\n" + "\n".join(lines) + "\n\n"

    events_text = json.dumps(candidate_events, indent=2)

    system_prompt = (
        "You are verifying soccer event detections from an automated pipeline. "
        "For each candidate event shown in the contact sheets, confirm or reject it "
        "based on what you see. A goal kick requires: ball near the 6-yard box, "
        "goalkeeper close to ball, opposing players pulled back. A corner kick requires: "
        "ball in the corner arc, players clustered in the penalty area. "
        "A throw-in requires: ball near the touchline, player about to throw. "
        "Halftime / empty field events should be rejected.\n\n"
        "Return ONLY valid JSON in this exact format:\n"
        '{"verified": [{"frame": <int>, "label": "<str>", "reason": "<str>"}], '
        '"rejected": [{"frame": <int>, "label": "<str>", "reason": "<str>"}]}'
    )

    content: list[dict] = []

    for sheet_path in sheet_paths:
        data = _encode_image(sheet_path)
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
        })

    content.append({
        "type": "text",
        "text": (
            f"{roster_text}"
            f"Candidate events to verify:\n{events_text}\n\n"
            "Review the contact sheets above and return the JSON verification result."
        ),
    })

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.content[0].text
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        import re
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        result = json.loads(m.group()) if m else {"verified": [], "rejected": []}

    result["raw_response"] = raw
    return result


def query_match(
    question: str,
    *,
    osl_path: Path | None = None,
    stats_path: Path | None = None,
    profile: dict | None = None,
    model: str = "claude-sonnet-4-6",
) -> str:
    """Answer a natural language question about a processed match.

    Sends OSL JSON + stats + optional profile as context.
    """
    client = _client()

    context_parts = []

    if osl_path and Path(osl_path).exists():
        with open(osl_path) as f:
            osl = json.load(f)
        context_parts.append(f"OSL events:\n{json.dumps(osl.get('events', []), indent=2)}")

    if stats_path and Path(stats_path).exists():
        with open(stats_path) as f:
            stats = json.load(f)
        context_parts.append(f"Match stats:\n{json.dumps(stats, indent=2)}")

    if profile:
        roster = profile.get("roster", [])
        roster_lines = [
            f"  #{p['jersey']} {p['name']} — {p.get('role', '')} — IDP: {p.get('idp_focus', '')}"
            for p in roster
        ]
        context_parts.append("Team profile:\n" + "\n".join(roster_lines))

    context = "\n\n".join(context_parts) if context_parts else "No match data provided."

    system_prompt = (
        "You are a soccer analysis assistant. Answer questions about match events, "
        "player performance, and tactics based on the structured data provided. "
        "Be concise and specific. Reference frame numbers and timestamps when relevant."
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": f"Match data:\n{context}\n\nQuestion: {question}",
        }],
    )

    return response.content[0].text
