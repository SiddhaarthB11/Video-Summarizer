"""Tests for the tolerant JSON parsing in agents.py -- no API key needed."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import parse_critic_json, parse_generated_references  # noqa: E402


def test_parse_critic_json_clean():
    r = parse_critic_json('{"critique": "needs more detail", "score": 0.7, "approved": false}')
    assert r.critique == "needs more detail"
    assert r.score == 0.7
    assert r.approved is False


def test_parse_critic_json_code_fenced():
    raw = '```json\n{"critique": "good", "score": 1.0, "approved": true}\n```'
    r = parse_critic_json(raw)
    assert r.approved is True
    assert r.score == 1.0


def test_parse_critic_json_clamps_score_out_of_range():
    r = parse_critic_json('{"critique": "x", "score": 1.7, "approved": false}')
    assert r.score == 1.0
    r2 = parse_critic_json('{"critique": "x", "score": -0.3, "approved": false}')
    assert r2.score == 0.0


def test_parse_critic_json_garbage_falls_back_gracefully():
    r = parse_critic_json("the model just said something not JSON at all")
    assert r.score == 0.0
    assert r.approved is False
    assert r.critique  # falls back to the raw text, not empty


def test_parse_generated_references_clean():
    raw = ('{"examples": [{"category": "cooking", "text": "A chef dices onions on a wooden board."},'
           '{"category": "cooking", "text": "A pot of pasta boils over on a stovetop."}]}')
    refs = parse_generated_references(raw)
    assert len(refs) == 2
    assert refs[0]["id"] == "ref-1"
    assert refs[0]["category"] == "cooking"


def test_parse_generated_references_skips_malformed_items():
    raw = '{"examples": [{"category": "x", "text": "A valid one."}, {"category": "y"}, "not a dict"]}'
    refs = parse_generated_references(raw)
    assert len(refs) == 1
    assert refs[0]["text"] == "A valid one."


def test_parse_generated_references_garbage_returns_empty():
    assert parse_generated_references("not json at all") == []
    assert parse_generated_references('{"nope": []}') == []
