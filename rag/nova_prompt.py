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
