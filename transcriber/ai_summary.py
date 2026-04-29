"""AI post-processing: turn a finished transcript into a structured
summary plus topic + action-item lists.

The single LLM call returns one JSON object so the UI doesn't have
to pay multiple round-trips to extract three deliverables. We use
the providers the app already depends on (OpenAI and Groq); both
SDKs implement ``response_format={"type": "json_object"}`` and the
same chat-completions shape, so the dispatch table can stay tiny.
"""

import json
import logging
from typing import Optional

from openai import OpenAI
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from .cloud_engine import _is_retriable_error

logger = logging.getLogger(__name__)


# Provider catalogue. The keys land in the sidebar dropdown verbatim,
# so the names are user-facing — keep them descriptive rather than
# matching the internal model IDs.
SUMMARY_PROVIDERS: dict[str, dict] = {
    "Groq (llama-3.3-70b)": {
        "client_factory": Groq,
        "model": "llama-3.3-70b-versatile",
        "description": "Fast, free tier available",
        # Same family as the transcription Groq option, so a user
        # already keyed in for transcription can reuse the credential.
        "matches_transcription_provider": "Groq",
    },
    "OpenAI (gpt-4o-mini)": {
        "client_factory": OpenAI,
        "model": "gpt-4o-mini",
        "description": "Slightly more accurate; pay-as-you-go",
        "matches_transcription_provider": "OpenAI",
    },
}


# The prompt is intentionally demanding: an underspecified instruction
# ("3-5 sentences", "a few topics") tends to produce work at the lower
# bound. Saying "exactly 3-7 short phrases", "all explicit action
# items", and constraining the JSON shape keeps the output usable by
# the renderer without a follow-up clean-up pass.
_PROMPT = """You are a senior editor producing structured metadata for a transcript.
Read the transcript below and respond with a single JSON object containing exactly these keys:

- "summary": a 3-5 sentence executive summary written in the same language as the transcript.
- "topics": an array of 3-7 short phrases (3-8 words each) describing the main subjects covered, ordered by importance.
- "action_items": an array of EVERY explicit action, decision, or commitment made in the transcript. Each item must be a complete sentence starting with a verb. When the transcript identifies the responsible person (a named speaker, "Speaker 0", a role, etc.) include the attribution at the start of the sentence. Use an empty array if no action items are present — never invent any.

Respond ONLY with the JSON object. No prose before or after.

Transcript:
{transcript}
"""


# Soft cap to protect users from accidentally sending a multi-hour
# transcript to a small-context model. Both supported models (llama-3.3
# at 128k tokens, gpt-4o-mini at 128k tokens) handle ~480k characters,
# so 380k leaves margin for the prompt scaffolding and answer.
_MAX_TRANSCRIPT_CHARS = 380_000


class SummaryError(RuntimeError):
    """Raised when the post-processing call fails or returns unusable data."""


def summarize_transcript(
    transcript: str,
    provider: str,
    api_key: str,
) -> dict:
    """Run an LLM over ``transcript`` and return ``{summary, topics, action_items}``.

    The tenacity retry mirrors the transcription path: only retry on
    transient errors so a bad API key fails fast instead of doing three
    rejected calls. Returns empty fields when the transcript is empty
    so callers don't have to special-case that.
    """
    if not transcript or not transcript.strip():
        return {"summary": "", "topics": [], "action_items": []}

    if provider not in SUMMARY_PROVIDERS:
        raise SummaryError(
            f"Unknown summary provider: {provider}. "
            f"Choose from: {list(SUMMARY_PROVIDERS)}"
        )

    if len(transcript) > _MAX_TRANSCRIPT_CHARS:
        raise SummaryError(
            f"Transcript is {len(transcript):,} characters — too long for "
            f"the configured summary model (max ~{_MAX_TRANSCRIPT_CHARS:,}). "
            "Shorten or split the transcript and try again."
        )

    config = SUMMARY_PROVIDERS[provider]
    client = config["client_factory"](api_key=api_key)
    return _call_with_retry(client, config["model"], transcript)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=_is_retriable_error,
    reraise=True,
)
def _call_with_retry(client, model: str, transcript: str) -> dict:
    """Issue the chat completion + parse the JSON response.

    Both OpenAI and Groq accept the same ``messages`` / ``temperature``
    / ``response_format`` shape, so we don't need a per-provider branch
    here — the only thing that differs is the underlying SDK class,
    which the caller already chose.
    """
    completion = client.chat.completions.create(
        model=model,
        # JSON mode means the model will refuse to output anything
        # other than a JSON document — without it we'd occasionally get
        # a polite "Sure, here's the summary:" prefix that breaks the
        # parser.
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You produce only valid JSON. Never wrap the JSON in code fences.",
            },
            {"role": "user", "content": _PROMPT.format(transcript=transcript)},
        ],
        # Low temperature for stability — summarisation isn't a
        # creative task and we want repeatable output across runs.
        temperature=0.2,
    )

    content = (completion.choices[0].message.content or "").strip()
    if not content:
        raise SummaryError("AI summary returned an empty response.")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        # Surface a short prefix so the user can see what came back
        # without flooding the UI with the full broken response.
        snippet = content[:300]
        logger.error("AI summary returned non-JSON: %r", snippet)
        raise SummaryError(
            f"AI summary returned invalid JSON ({exc}). Try regenerating."
        ) from exc

    return _normalise_summary_payload(data)


def _normalise_summary_payload(data: object) -> dict:
    """Coerce the LLM's JSON object into the exact shape callers expect.

    Models occasionally return slight schema drift (a string instead of
    a list, a single-item list as a string, etc.); normalising here
    means the renderer doesn't need defensive isinstance() checks at
    every render call.
    """
    if not isinstance(data, dict):
        raise SummaryError(
            f"AI summary returned a {type(data).__name__}, expected a JSON object."
        )

    summary = str(data.get("summary") or "").strip()
    topics = _coerce_string_list(data.get("topics"))
    action_items = _coerce_string_list(data.get("action_items"))

    return {
        "summary": summary,
        "topics": topics,
        "action_items": action_items,
    }


def _coerce_string_list(value: object) -> list[str]:
    """Best-effort convert ``value`` into a list of trimmed strings.

    Accepts:
      * ``None`` / missing → ``[]``
      * a list/tuple → keep order, drop empties
      * a single string → split on newlines (the model occasionally
        returns a bullet list as a single string when it fights JSON
        mode)
    """
    if value is None:
        return []
    if isinstance(value, str):
        # Split on newlines, strip leading bullet/dash markers.
        lines = [line.strip().lstrip("-•* ").strip() for line in value.splitlines()]
        return [line for line in lines if line]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    # Anything else (number, dict, etc.) → empty rather than raising.
    return []
