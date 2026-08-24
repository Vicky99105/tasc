"""One POST to an incoming webhook, after approval only. The webhook URL is bound
to one channel at creation, so there is no channel-selection code and no way to
post somewhere unintended. Never logged, never printed.
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Protocol


class SlackClient(Protocol):
    def post(self, text: str) -> str: ...


def build_slack_payload(markdown: str) -> dict:
    lines = markdown.strip().split("\n")
    role_header = "Candidate Shortlist"
    for l in lines:
        if l.startswith("# "):
            role_header = l[2:].strip()
            break

    parts = markdown.split("\n## ")
    candidate_parts = parts[1:] if len(parts) > 1 else []

    overview_lines = []
    attachments = []

    for part in candidate_parts:
        clines = part.strip().split("\n")
        title_line = clines[0]  # e.g. "C088 — 75.3"
        cid = title_line.split("—")[0].strip()
        score = title_line.split("—")[1].strip() if "—" in title_line else ""

        headline = ""
        summary = ""
        diff = ""
        strengths = []
        experience = ""
        questions = []

        mode = "body"
        for line in clines[1:]:
            sline = line.strip()
            if not sline or sline == "---":
                continue
            if sline.startswith("_") and sline.endswith("_") and not headline:
                headline = sline.strip("_")
            elif sline.startswith("**What sets them apart:**"):
                diff = sline.replace("**What sets them apart:**", "").strip()
            elif sline.startswith("**Strengths**"):
                mode = "strengths"
            elif sline.startswith("**Relevant experience:**"):
                experience = sline.replace("**Relevant experience:**", "").strip()
                mode = "body"
            elif sline.startswith("**Questions**"):
                mode = "questions"
            elif mode == "strengths" and sline.startswith("- "):
                strengths.append(sline[2:])
            elif mode == "questions" and re.match(r"^\d+\.", sline):
                questions.append(sline)
            elif mode == "body" and not sline.startswith("**") and not sline.startswith("_"):
                if not summary:
                    summary = sline

        strength_summary = ", ".join(s.split("—")[0].strip() for s in strengths[:3])
        exp_summary = experience.split("—")[0].strip() if experience else "N/A"
        overview_lines.append(
            f"• *{cid}* (Score: `{score}`) | _{headline}_ | *Exp:* {exp_summary} | *Top:* {strength_summary or 'Core fit'}"
        )

        att_text_parts = []
        if diff:
            att_text_parts.append(f"*Key Differentiator:* {diff}")
        if strengths:
            att_text_parts.append("*Strengths:*\n" + "\n".join(f"• {s}" for s in strengths))
        if experience:
            att_text_parts.append(f"*Experience:* {experience}")
        if questions:
            att_text_parts.append("*Suggested Screening Questions:*\n" + "\n".join(questions))

        attachments.append({
            "color": "#0f766e",
            "title": f"{cid} — Score: {score} ({headline})",
            "text": "\n\n".join(att_text_parts),
            "mrkdwn_in": ["text", "title"]
        })

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"🎯 Shortlist: {role_header}", "emoji": True}
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Finalists Overview:*\n" + ("\n".join(overview_lines) if overview_lines else "No candidates cleared the threshold.")
            }
        },
        {"type": "divider"}
    ]

    return {
        "text": f"🎯 Candidate Shortlist: {role_header} ({len(candidate_parts)} Finalists)",
        "blocks": blocks,
        "attachments": attachments
    }


class DefaultSlackClient:
    def __init__(self, webhook_url: str):
        self._webhook_url = webhook_url

    def post(self, text: str) -> str:
        payload = build_slack_payload(text) if ("\n## " in text or text.startswith("# ")) else {"text": text}
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._webhook_url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            status = resp.status
            reply = resp.read().decode()
        if status != 200:
            raise RuntimeError(f"slack post failed: {status} {reply}")
        return reply  # incoming webhooks return "ok", not a real message ts


class FakeSlackClient:
    def __init__(self):
        self.posts: list[str] = []

    def post(self, text: str) -> str:
        self.posts.append(text)
        return "ok"

