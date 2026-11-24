"""Nova's spoken-answer prompts and page helpers, shared by the voice bridge and image tests.

These are NOT the frozen benchmark prompts (P01-P03 stay untouched in the manifest).
"""
import re

NOT_FOUND_REPLY = "I couldn't find that in the manual."

_RULES = (
    "Rules:\n"
    "- Reply in 1 to 3 short sentences that sound natural when read aloud. "
    "No lists, tables, markdown or symbols.\n"
    "- Write units as words (millimetres, litres, kilowatts).\n"
    "- Mention the page number, for example \"According to page 11, ...\". Use the [Page N] "
    "label of the excerpt you actually took the answer from, not the label of another "
    "excerpt.\n"
    "- You may do simple arithmetic (add, subtract, multiply, divide, convert units) on "
    "numbers from the manual, and give the result.\n"
    f"- If the excerpts do not contain the answer, reply exactly: {NOT_FOUND_REPLY}\n\n"
)

# Text answer with qwen3 (voice questions, and image Mode A after the vision description)
NOVA_PROMPT = {
    "system_prompt": (
        "You are Nova, a voice assistant for the SV-CH / SV-F universal milling machine.\n"
        "Answer the question using only the manual excerpts below. Each excerpt starts with a "
        "[Page N] label.\n" + _RULES + "Manual excerpts:\n{context}"
    ),
    "user_template": "{question}",
    "not_found": NOT_FOUND_REPLY,
}

# Image Mode A, step 1: qwen2.5vl describes the image (no manual context)
DESCRIBE_SYSTEM = (
    "You look at a photo of a universal milling machine (SV-CH / SV-F), one of its parts, or a "
    "page of its manual. The user's question is given for context; do not answer it.\n"
    "Reply in this format:\n"
    "DESCRIPTION: 1 or 2 short sentences on what is shown.\n"
    "VISIBLE TEXT: any readable text, labels, part names, codes and numbers, copied exactly, "
    "separated by semicolons. Write \"none\" if there is none.\n"
    "Only report what you can actually see. Do not guess text you cannot read."
)
# qwen2.5vl follows the user turn much more than the system turn, so the task is repeated here.
DESCRIBE_USER = (
    "Do not answer this question, it is only context: \"{question}\"\n\n"
    "Describe the image instead, in exactly this format:\n"
    "DESCRIPTION: <1 or 2 sentences on what is shown>\n"
    "VISIBLE TEXT: <readable text, labels, part names, codes, numbers; separated by semicolons>"
)

# Image Mode A, step 2: what qwen3 gets as the "question" (it never sees the image itself)
MODE_A_QUESTION = "{question}\n\n(What the camera sees, described automatically: {description})"

# Image Mode B: qwen2.5vl answers from image + question + retrieved excerpts in one call
NOVA_VISION_PROMPT = (
    "You are Nova, a voice assistant for the SV-CH / SV-F universal milling machine.\n"
    "The user sent an image with their question. Use the image only to understand what they "
    "are looking at. Take every fact in your answer from the manual excerpts below, never from "
    "your own knowledge. Each excerpt starts with a [Page N] label.\n"
    + _RULES + "Manual excerpts:\n{context}"
)


# Image Mode B2: same flow as B, stricter spoken/page/honesty rules. Mode B's prompt above is
# kept unchanged for the record. qwen2.5vl weights the user turn most, so the key rules are
# repeated there (B2_USER).
NOVA_VISION_PROMPT_B2 = (
    "You are Nova, a voice assistant for the SV-CH / SV-F universal milling machine.\n"
    "The user sent an image with a question. Use the image only to see what they are looking "
    "at. Every fact in your answer must come from the manual excerpts below. Never use general "
    "knowledge. Each excerpt starts with a [Page N] label.\n"
    "Rules:\n"
    "- Answer in 1 to 3 short sentences that sound natural when read aloud. Never use lists, "
    "numbered steps, bullet points, headings, markdown or symbols.\n"
    "- Write units as words (millimetres, litres, kilowatts).\n"
    "- You may do simple arithmetic on numbers from the manual.\n"
    "- End your answer with the page you used, written as \"Page N\", where N comes from the "
    "[Page N] label of the excerpt that contains the answer.\n"
    "- If the excerpts contain the answer, give it even if the wording differs from the "
    "question.\n"
    "- If the image does not show this milling machine or its manual, or the question is too "
    "vague, or the excerpts do not contain the answer, reply exactly: "
    f"{NOT_FOUND_REPLY}\n\n"
    "Manual excerpts:\n{context}"
)
B2_USER = (
    "{question}\n\n"
    "(Answer in 1 to 3 spoken sentences using only the manual excerpts, no lists, and end with "
    f"\"Page N\". If the excerpts do not answer it, say exactly: {NOT_FOUND_REPLY})"
)
# v1 = first B2 run (commit 39f8821); v2 adds the "wording differs" line (Q028 fix).
B2_PROMPT_ID = "NOVA_IMG_B2_v2"
# Mode B2 skips the vision model when retrieval is too weak (same score rule as the voice route)
B2_LOW_SCORE_REPLY = f"{NOT_FOUND_REPLY} Could you ask about a specific part or setting?"

# Camera questions (/live, ESP), describe step: ONE vision call gives (a) a short spoken answer
# from the image only, used directly when the manual has no answer (like the voice route's chat
# fallback: no manual facts, no page numbers), and (b) the visible text, for retrieval.
LOOK_DESCRIBE_SYSTEM = (
    "You are Nova, a voice assistant with a camera, looking at the user's camera image.\n"
    "Reply in exactly this format:\n"
    "SPOKEN: one or two short sentences that answer the user's question from what you can see "
    "(or say what the image shows), natural when read aloud, for example: \"I see a person at a "
    "desk with a laptop.\" Never say that you can't see or have no camera: if what they ask "
    "about is not in view, say so briefly (\"There's no lever in view.\"; only if they ask "
    "about themselves: \"You're not in view right now.\") and say what you do see. Never "
    "mention a manual, page numbers, specifications, capacities, part numbers or "
    "procedures, and never give operating or safety instructions.\n"
    "VISIBLE TEXT: readable text, labels, part names and numbers, copied exactly, separated by "
    "semicolons, or \"none\".\n"
    "Only report what you can actually see. Do not guess text you cannot read."
)
LOOK_DESCRIBE_USER = "{question}\n\n(Reply as SPOKEN: ... then VISIBLE TEXT: ...)"
# Added when the question or the image is about the machine: the manual had no answer.
# ("I couldn't find" rather than "isn't covered": a vague question about a real manual page can
# miss in retrieval although the page is in the manual.)
GENERAL_NOT_IN_MANUAL = NOT_FOUND_REPLY
MACHINE_RE = re.compile(
    r"\b(milling|machines?|manual|spindle|coolant|lubricat\w*|levers?|handwheels?|gears?|arbor|"
    r"quill|knee|saddle|vise|vice|cutters?|slideways?|diagrams?|schematics?)\b", re.I)


# Spoken questions about what the camera sees (/live, ESP) -> image route. Phrase-based: bare
# "see" / "look" are not enough ("See you", "And you see the machine here", "Look, what is ...").
# SEE_RE: only the camera can answer ("Can you see me?") -> without a frame Nova says it couldn't
#         get a picture (never "I can't see").
# REF_RE: points at something ("What is this?", "this lever") -> the camera if a frame is there,
#         else the normal voice route (the manual may still answer "How often should this lever be oiled?").
_PARTS = (r"machine|part|thing|lever|button|handle|handwheel|switch|knob|component|piece|plate|"
          r"label|sign|page|tool|device|object|one|screen|panel|gauge|diagram")
SEE_RE = re.compile(
    r"\bwhat (?:do|can|did|could) you see\b"
    r"|\b(?:do|can|could) you (?:still )?see (?:me|it|this|that|these|those|anything|what|my|"
    r"the|a|an|any)\b"
    r"|\b(?:look|looking) at (?:this|that|these|those|it|me|my|the)\b"
    r"|\b(?:take|have) a look\b"
    r"|\bwhat am i (?:holding|pointing at|pointing to|showing you|looking at)\b"
    r"|\bwhat are you looking at\b"
    r"|\b(?:identify|describe) (?:this|that|it|what you see)\b"
    r"|\bin front of (?:me|you|the camera)\b|\b(?:on|in|through) the camera\b"
    r"|\bshow(?:ing)? you\b",
    re.I)
REF_RE = re.compile(
    r"\bwhat(?:'s| is| are) (?:this|that|these|those)\b"
    r"(?!\s+(?:error|code|alarm|warning|message|word|term|number|value)s?\b)"
    rf"|\b(?:this|that|these|those) (?:{_PARTS})s?\b",
    re.I)


def _norm(text):
    return text.replace("’", "'")


def is_look_question(text):
    """Camera question: SEE_RE or REF_RE."""
    t = _norm(text)
    return bool(SEE_RE.search(t) or REF_RE.search(t))


def needs_camera(text):
    """Only the camera can answer (SEE_RE): no frame -> "couldn't get a picture", not chat."""
    return bool(SEE_RE.search(_norm(text)))


# Does the IMAGE (its spoken description / visible text) look like the machine or its manual?
# Only then is a second manual attempt worth a vision call (camera answers must stay fast).
IMAGE_RELATED_RE = re.compile(
    MACHINE_RE.pattern + r"|\b(documents?|pages?|specifications?|technical|instructions?|"
    r"labels?|tables?|parts?)\b", re.I)


def strip_page_mentions(text):
    """Drop any sentence that names a page (general answers must not cite the manual)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [s for s in sentences if not re.search(r"\bpages?\s+\d+", s, re.I)]
    return " ".join(kept).strip()


def label_pages(hits):
    """'[Page 51 - 7-4 Section]' first line -> '[Page 51]' + 'Section: 7-4 Section'."""
    out = []
    for h in hits:
        text = re.sub(r"^\[Page ([\d-]+) - (.*?)\]\s*", r"[Page \1]\nSection: \2\n", h["text"])
        out.append({**h, "text": text})
    return out


def mentioned_pages(answer):
    """Page numbers the answer names ("page 11", "pages 12")."""
    return [int(p) for p in re.findall(r"\bpages?\s+(\d+)", answer, re.I)]


def answer_pages(answer, retrieved_pages, not_found):
    """Pages to report: the cited ones that were really retrieved, else all retrieved pages."""
    if not_found:
        return []
    return [p for p in mentioned_pages(answer) if p in retrieved_pages] or list(retrieved_pages)


def is_not_found(answer):
    norm = lambda s: s.upper().replace("’", "'")
    return "NOT FOUND" in norm(answer) or norm(NOT_FOUND_REPLY) in norm(answer)
