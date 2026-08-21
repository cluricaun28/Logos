"""Regression tests for the F3 single-word topic filter (2026-08-17).

The session-end extractor's Pattern-1 regex captures any Capitalized word,
which historically produced 42K junk topics ('Good' x170, 'Please' x118,
'None' x127 — sentence starters and Python/JSON literals from tool output).
The _JUNK_SINGLE_WORDS filter rejects junk single words while preserving
multi-word phrases, CamelCase identifiers, file references, and domain
nouns.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.memory.perpetual_context.session_end_extractor import (
    _JUNK_SINGLE_WORDS,
    extract_topics_from_messages,
)

STOPWORDS = frozenset(
    {"the", "and", "this", "that", "with", "from", "have", "been", "is", "it"}
)


def _msgs(*texts):
    return [{"role": "user", "content": t} for t in texts]


class TestSingleWordJunkFilter:
    def test_sentence_starters_rejected(self):
        result = extract_topics_from_messages(
            _msgs("Good. Please check the logs. These files are ready."), STOPWORDS
        )
        for junk in ("Good", "Please", "These"):
            assert junk not in result

    def test_python_literals_rejected(self):
        result = extract_topics_from_messages(
            _msgs("The value is None. The flag is True. Status: False."),
            STOPWORDS,
        )
        for junk in ("None", "True", "False"):
            assert junk not in result

    def test_status_and_generic_nouns_rejected(self):
        result = extract_topics_from_messages(
            _msgs("Current status: Updated. Tool output shows an error."),
            STOPWORDS,
        )
        for junk in ("Current", "Status", "Updated", "Tool", "Error"):
            assert junk not in result

    def test_domain_nouns_preserved(self):
        result = extract_topics_from_messages(
            _msgs("Working on the Logos Gateway today. Telegram polling tested."),
            STOPWORDS,
        )
        assert "Logos Gateway" in result
        assert "Telegram" in result

    def test_multiword_phrases_preserved(self):
        result = extract_topics_from_messages(
            _msgs("Discussed Rolling Window behavior and Docker Networking."),
            STOPWORDS,
        )
        # Pattern 1 greedily captures up to 3 capitalized words, so the
        # phrase may be joined with its sentence opener — check substring.
        assert any("Rolling Window" in t for t in result)
        assert "Docker Networking" in result


class TestOtherPatternsUnaffected:
    def test_camelcase_identifiers_preserved(self):
        result = extract_topics_from_messages(
            _msgs("Fixed the SemanticVectorContextEngine fallback path."),
            STOPWORDS,
        )
        assert "SemanticVectorContextEngine" in result

    def test_file_references_preserved(self):
        result = extract_topics_from_messages(
            _msgs("Changes went into config.yaml and the restart script."),
            STOPWORDS,
        )
        assert "config.yaml" in result

    def test_junk_set_has_no_duplicates_or_typos(self):
        # Guards against copy/paste accidents like a doubled entry.
        assert len(_JUNK_SINGLE_WORDS) == len(set(_JUNK_SINGLE_WORDS))
        assert all(w == w.strip() and w for w in _JUNK_SINGLE_WORDS)
        assert "stop2" not in _JUNK_SINGLE_WORDS
        assert "stopped2" not in _JUNK_SINGLE_WORDS
