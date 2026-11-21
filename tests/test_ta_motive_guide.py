"""Variant B's motive (the Professor did not renew the Teaching Assistant's contract) is on no card:
only the Professor says it, when asked about the TA. A blind playtest never asked and had to guess
the "why". The guide now suggests that question once every card is found, in every variant."""

import pytest

from game.case import load_case
from game.engine import GUIDE_QUESTIONS, Engine
from tests.test_guide_and_proofs import follow_the_guide

CASE = load_case()


def guided_game(variant, mode="normal"):
    engine = Engine()
    game = engine.new_game(seed=1, variant=variant)
    if mode == "hard":
        engine.set_mode(game, mode)
    return engine, game


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_the_guide_asks_the_professor_about_the_ta_before_the_notebook(variant):
    engine, game = guided_game(variant)
    texts = follow_the_guide(engine, game)
    assert "Ask the Professor about the Teaching Assistant." in texts
    assert texts.index("Ask the Professor about the Teaching Assistant.") == len(texts) - 2
    assert texts[-1].startswith("You have every card")
    assert any(e["action"] == "ask" and e["suspect"] == "professor" and e["topic"] == "teaching_assistant"
               for e in game.history)


def test_in_variant_b_that_question_tells_the_motive_and_elsewhere_it_does_not():
    for variant, motive in (("A", False), ("B", True), ("C", False)):
        engine, game = guided_game(variant)
        line = engine.ask(game, "professor", "teaching_assistant")["line"]
        assert ("renew their contract" in line) is motive, variant


def test_the_step_is_skipped_once_the_question_was_asked():
    engine, game = guided_game("A")
    engine.ask(game, "professor", "teaching_assistant")
    assert "Ask the Professor about the Teaching Assistant." not in follow_the_guide(engine, game)


def test_the_step_explains_why_and_the_why_never_names_a_culprit():
    engine2, game2 = guided_game("B")
    while True:
        step = engine2.next_step(game2)
        if step["do"] == "ask" and step["topic"] == "teaching_assistant":
            break
        {"search": lambda: engine2.open_clue(game2, step["clue"]),
         "show": lambda: engine2.present(game2, step["suspect"], step["clue"]),
         "ask": lambda: engine2.ask(game2, step["suspect"], step["topic"])}[step["do"]]()
    assert step["why"] == GUIDE_QUESTIONS[0]["why"]
    assert engine2.thought(game2, step) == step["why"]
    assert "leak" not in step["why"].lower() and "contract" not in step["why"].lower()


def test_hard_mode_still_has_no_guide():
    engine, game = guided_game("B", mode="hard")
    assert engine.next_step(game) is None


def test_the_guide_is_still_the_same_in_every_variant():
    paths = {v: follow_the_guide(*guided_game(v)) for v in "ABC"}
    assert paths["A"] == paths["B"] == paths["C"]
