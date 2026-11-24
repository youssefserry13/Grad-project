"""Unit test: which spoken questions go to the camera (image) route.

    python test_look_question.py        (or: pytest test_look_question.py)
"""
from nova_prompt import is_look_question, needs_camera

LOOK = [
    "What do you see?",
    "What can you see on the table?",
    "Look at this and tell me what it is.",
    "What is this?",
    "What's that thing on the left?",
    "What am I holding?",
    "What am I pointing at?",
    "What is this part used for?",
    "How often should this lever be oiled?",
    "Can you see me?",
    "Can you see the machine?",
    "Do you see anything?",
    "What's in front of you?",
    "Let me show you something.",
]
# camera-only questions: without a frame Nova says it couldn't get a picture (never "can't see")
CAMERA_ONLY = ["Can you see me?", "Can you see the machine?", "What's in front of you?",
               "What am I holding?", "Let me show you something."]
# point at something: without a frame the voice route (manual) still answers
NOT_CAMERA_ONLY = ["How often should this lever be oiled?", "What is this part used for?"]
NOT_LOOK = [
    "See you.",
    "And you see the machine here",
    "See you later, Nova.",
    "Look, what is the capacity of the lubrication tank?",
    "I'll look into the coolant problem later.",
    "What is that error code E21?",
]


def test_look():
    assert all(is_look_question(q) for q in LOOK), [q for q in LOOK if not is_look_question(q)]


def test_camera_only():
    assert all(needs_camera(q) for q in CAMERA_ONLY), [q for q in CAMERA_ONLY if not needs_camera(q)]
    assert not any(needs_camera(q) for q in NOT_CAMERA_ONLY)


def test_not_look():
    wrong = [q for q in NOT_LOOK if is_look_question(q)]
    assert not wrong, wrong


if __name__ == "__main__":
    for q in LOOK + NOT_LOOK:
        print(f"{'LOOK    ' if is_look_question(q) else 'not look'}  {q}")
    test_look()
    test_not_look()
    test_camera_only()
    print(f"OK: {len(LOOK)} look + {len(NOT_LOOK)} not-look sentences, "
          f"{len(CAMERA_ONLY) + len(NOT_CAMERA_ONLY)} camera-only checks")
