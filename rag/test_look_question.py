"""Unit test: which spoken questions go to the camera (image) route.

    python test_look_question.py        (or: pytest test_look_question.py)
"""
from nova_prompt import is_look_question

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
]
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


def test_not_look():
    wrong = [q for q in NOT_LOOK if is_look_question(q)]
    assert not wrong, wrong


if __name__ == "__main__":
    for q in LOOK + NOT_LOOK:
        print(f"{'LOOK    ' if is_look_question(q) else 'not look'}  {q}")
    test_look()
    test_not_look()
    print(f"OK: {len(LOOK)} look + {len(NOT_LOOK)} not-look sentences")
