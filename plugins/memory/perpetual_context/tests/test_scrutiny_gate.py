"""Tests for Scrutiny Gate — Phase 3 of the Deep Research & Continuity Engine.

Verifies:
1. TopicSensitivityClassifier correctly classifies technical vs political queries
2. ScrutinyGate.vet_results() returns structured results with warnings
3. detect_bias() identifies loaded language patterns
4. RLIngestionGate.evaluate() flags contradictions for manual review
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

# Ensure the perpetual_context package is importable
sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent/plugins/memory"))

from perpetual_context.scrutiny_gate import (
    BIAS_CONFIDENCE_THRESHOLD,
    DEFAULT_SENSITIVITY,
    RL_CONTRADICTION_THRESHOLD,
    ScrutinyGate,
    TopicSensitivityClassifier,
    RLIngestionGate,
    _extract_domain,
    _get_source_stance,
)


class TestTopicSensitivityClassifier(unittest.TestCase):
    """Test topic sensitivity classification."""

    def setUp(self):
        self.classifier = TopicSensitivityClassifier()

    # --- Low sensitivity queries ---

    def test_classify_python_code_query_as_low(self):
        result = self.classifier.classify("How to write a Python decorator")
        self.assertEqual(result, "low")

    def test_classify_docker_deploy_query_as_low(self):
        result = self.classifier.classify("Docker compose deploy to kubernetes cluster")
        self.assertEqual(result, "low")

    def test_classify_gpu_hardware_query_as_low(self):
        result = self.classifier.classify("RTX 5090 GPU memory bandwidth specs")
        self.assertEqual(result, "low")

    def test_classify_database_sql_query_as_low(self):
        result = self.classifier.classify("PostgreSQL database indexing and SQL optimization")
        self.assertEqual(result, "low")

    def test_classify_build_compile_debug_as_low(self):
        result = self.classifier.classify("How to build and compile a C++ project with debug symbols")
        self.assertEqual(result, "low")

    # --- High sensitivity queries ---

    def test_classify_politics_query_as_high(self):
        result = self.classifier.classify("US election results 2024 analysis")
        self.assertEqual(result, "high")

    def test_classify_history_query_as_high(self):
        result = self.classifier.classify("History of the Cold War and its impact")
        self.assertEqual(result, "high")

    def test_classify_climate_query_as_high(self):
        result = self.classifier.classify("Climate change policy debate in Congress")
        self.assertEqual(result, "high")

    def test_classify_religion_query_as_high(self):
        result = self.classifier.classify("Christianity theology and faith practices")
        self.assertEqual(result, "high")

    def test_classify_race_gender_identity_as_high(self):
        result = self.classifier.classify("Gender identity and social justice movements")
        self.assertEqual(result, "high")

    def test_classify_war_military_query_as_high(self):
        result = self.classifier.classify("Military defense strategy in the Pacific")
        self.assertEqual(result, "high")

    # --- Edge cases ---

    def test_classify_empty_string_defaults_to_high(self):
        result = self.classifier.classify("")
        self.assertEqual(result, DEFAULT_SENSITIVITY)
        self.assertEqual(result, "high")

    def test_classify_none_returns_default_sensitivity(self):
        result = self.classifier.classify(None)  # type: ignore[arg-type]
        self.assertEqual(result, DEFAULT_SENSITIVITY)

    def test_classify_non_string_returns_default(self):
        result = self.classifier.classify(12345)  # type: ignore[arg-type]
        self.assertEqual(result, DEFAULT_SENSITIVITY)

    def test_classify_ambiguous_query_defaults_to_high(self):
        """A query with no clear keywords should default to high (safer)."""
        result = self.classifier.classify("What happened yesterday in the city")
        self.assertEqual(result, "high")

    def test_classify_mixed_technical_and_political_is_high(self):
        """High-sensitivity keywords take priority over low."""
        result = self.classifier.classify("Python code for election data analysis")
        # Contains both 'python' (low) and 'election' (high) — high wins
        self.assertEqual(result, "high")

    def test_classify_case_insensitive(self):
        """Classification should be case-insensitive."""
        result = self.classifier.classify("POLITICS AND GOVERNMENT POLICY")
        self.assertEqual(result, "high")


class TestScrutinyGate(unittest.TestCase):
    """Test the main scrutiny orchestrator."""

    def setUp(self):
        self.gate = ScrutinyGate()

    # --- vet_results tests ---

    def test_vet_results_returns_structured_output(self):
        results = [
            {
                "title": "Python Tutorial",
                "url": "https://docs.python.org/3/tutorial/",
                "snippet": "Learn Python programming basics.",
            }
        ]
        output = self.gate.vet_results(results, query="python tutorial")

        self.assertIn("vetted_results", output)
        self.assertIn("rejected_results", output)
        self.assertIn("sensitivity_level", output)
        self.assertIn("warnings", output)
        self.assertEqual(output["sensitivity_level"], "low")
        self.assertEqual(len(output["vetted_results"]), 1)

    def test_vet_results_empty_input(self):
        output = self.gate.vet_results([], query="anything")
        self.assertEqual(output["vetted_results"], [])
        self.assertIn("No results to vet", output["warnings"])

    def test_vet_results_rejects_empty_result(self):
        results = [{}]  # Empty dict — no title, url, or snippet
        output = self.gate.vet_results(results, query="test")
        self.assertEqual(len(output["rejected_results"]), 1)
        self.assertIn("Empty result", output["rejected_results"][0]["rejection_reason"])

    def test_vet_results_high_sensitivity_includes_bias_analysis(self):
        results = [
            {
                "title": "Political Analysis",
                "url": "https://www.nytimes.com/politics/article",
                "snippet": "Experts say the policy is beneficial for all citizens.",
            }
        ]
        output = self.gate.vet_results(results, query="US politics election policy")

        self.assertEqual(output["sensitivity_level"], "high")
        vetted = output["vetted_results"][0]
        metadata = vetted.get("scrutiny_metadata", {})
        self.assertIn("bias_analysis", metadata)
        self.assertIn("source_stance", metadata)

    def test_vet_results_low_sensitivity_basic_validation(self):
        results = [
            {
                "title": "Docker Guide",
                "url": "https://docs.docker.com/get-started/",
                "snippet": "Get started with Docker containers.",
            }
        ]
        output = self.gate.vet_results(results, query="docker container setup")

        self.assertEqual(output["sensitivity_level"], "low")
        vetted = output["vetted_results"][0]
        metadata = vetted.get("scrutiny_metadata", {})
        # Low sensitivity should have minimal bias analysis
        self.assertFalse(metadata["bias_analysis"].get("detected", False))

    def test_vet_results_preserves_original_fields(self):
        original = {
            "title": "Test Article",
            "url": "https://example.com/article",
            "snippet": "Some content here.",
            "extra_field": "should survive",
        }
        output = self.gate.vet_results([original], query="test")
        vetted = output["vetted_results"][0]
        self.assertEqual(vetted["title"], original["title"])
        self.assertEqual(vetted["url"], original["url"])
        self.assertEqual(vetted["extra_field"], "should survive")

    # --- detect_bias tests ---

    def test_detect_bias_identifies_loaded_language(self):
        text = (
            "The so-called expert allegedly made toxic claims that were "
            "shockingly disgraceful and scandalous."
        )
        report = self.gate.detect_bias(text)

        self.assertTrue(report["detected"])
        self.assertGreater(len(report["loaded_language"]), 0)
        # Should find multiple loaded words
        self.assertIn("so-called", report["loaded_language"])
        self.assertIn("allegedly", report["loaded_language"])
        self.assertIn("toxic", report["loaded_language"])

    def test_detect_bias_identifies_passive_voice(self):
        text = "Mistakes were made and errors were found in the report."
        report = self.gate.detect_bias(text)

        self.assertGreater(report["passive_voice_count"], 0)

    def test_detect_bias_identifies_vague_attribution(self):
        text = "Experts say the policy is harmful. Studies show it fails."
        report = self.gate.detect_bias(text)

        self.assertGreater(report["vague_attribution_count"], 0)

    def test_detect_bias_neutral_text_returns_low_confidence(self):
        text = (
            "The Python programming language was released in 1991 by Guido van Rossum. "
            "It supports multiple programming paradigms including procedural and object-oriented."
        )
        report = self.gate.detect_bias(text)

        # Neutral technical content should have low or no bias detection
        self.assertFalse(report["detected"])
        self.assertEqual(len(report["loaded_language"]), 0)

    def test_detect_bias_source_stance_mapping(self):
        report_left = self.gate.detect_bias("Some text", "https://www.nytimes.com/article")
        self.assertEqual(report_left["source_stance"], "center-left")

        report_right = self.gate.detect_bias("Some text", "https://www.wsj.com/article")
        self.assertEqual(report_right["source_stance"], "center-right")

        report_centrist = self.gate.detect_bias("Some text", "https://www.reuters.com/article")
        self.assertEqual(report_centrist["source_stance"], "centrist")

    def test_detect_bias_empty_text_returns_neutral(self):
        report = self.gate.detect_bias("")
        self.assertFalse(report["detected"])
        self.assertEqual(report["max_confidence"], 0.0)

    def test_detect_bias_none_text_returns_neutral(self):
        report = self.gate.detect_bias(None)  # type: ignore[arg-type]
        self.assertFalse(report["detected"])

    def test_detect_bias_non_string_returns_neutral(self):
        report = self.gate.detect_bias(12345)  # type: ignore[arg-type]
        self.assertFalse(report["detected"])

    # --- apply_worldview_filter tests ---

    def test_worldview_filter_annotates_facts(self):
        facts = [
            {"content": "Python is a programming language.", "source": "https://python.org"},
        ]
        result = self.gate.apply_worldview_filter(facts, query="python basics")

        self.assertEqual(len(result), 1)
        fact = result[0]
        self.assertIn("scrutiny_level", fact)
        self.assertIn("source_bias_note", fact)
        self.assertIn("worldview_alignment", fact)
        self.assertIn("confidence", fact)

    def test_worldview_filter_high_sensitivity_deep_vetting(self):
        facts = [
            {"content": "The policy was beneficial for all citizens.", "source": "https://www.nytimes.com"},
        ]
        result = self.gate.apply_worldview_filter(facts, query="US politics election")

        fact = result[0]
        self.assertEqual(fact["scrutiny_level"], "deep")

    def test_worldview_filter_low_sensitivity_basic_vetting(self):
        facts = [
            {"content": "Docker containers are lightweight.", "source": "https://docs.docker.com"},
        ]
        result = self.gate.apply_worldview_filter(facts, query="docker containers")

        fact = result[0]
        self.assertEqual(fact["scrutiny_level"], "basic")

    def test_worldview_filter_empty_facts_returns_empty(self):
        result = self.gate.apply_worldview_filter([], query="anything")
        self.assertEqual(result, [])

    def test_worldview_filter_does_not_alter_original_content(self):
        original_content = "The original fact content."
        facts = [{"content": original_content}]
        result = self.gate.apply_worldview_filter(facts, query="test")
        self.assertEqual(result[0]["content"], original_content)


class TestRLIngestionGate(unittest.TestCase):
    """Test Reference Library ingestion gating."""

    def setUp(self):
        self.gate = RLIngestionGate()

    # --- Basic evaluation tests ---

    def test_evaluate_empty_content_rejected(self):
        result = self.gate.evaluate({"content": ""})
        self.assertEqual(result["status"], "rejected")
        self.assertIn("Empty content", result["reason"])

    def test_evaluate_missing_content_rejected(self):
        result = self.gate.evaluate({})
        self.assertEqual(result["status"], "rejected")

    def test_evaluate_returns_structured_output(self):
        result = self.gate.evaluate({
            "content": "Python 3.12 was released in October 2023.",
            "url": "https://www.python.org/downloads/",
            "query": "python release date",
        })

        self.assertIn("status", result)
        self.assertIn("reason", result)
        self.assertIn("contradictions", result)
        self.assertIn("source_assessment", result)
        self.assertIn("scrutiny_passed", result)

    def test_evaluate_approved_for_low_sensitivity(self):
        """Technical content from neutral source should be approved."""
        result = self.gate.evaluate({
            "content": "Docker uses Linux namespaces and cgroups for isolation.",
            "url": "https://docs.docker.com/engine/",
            "query": "docker architecture",
        })
        # Should be approved — low sensitivity, neutral source
        self.assertEqual(result["status"], "approved")

    def test_evaluate_manual_review_for_high_bias(self):
        """High-sensitivity content with strong bias should go to manual review."""
        result = self.gate.evaluate({
            "content": (
                "The so-called expert allegedly made toxic and scandalous claims. "
                "Experts say the policy is disgraceful. Reports indicate it was "
                "shockingly harmful. Studies show mistakes were made everywhere."
            ),
            "url": "https://www.breitbart.com/politics/",
            "query": "US politics election",
        })
        # High sensitivity + high bias = manual review
        self.assertEqual(result["status"], "manual_review")

    def test_evaluate_source_assessment_populated(self):
        result = self.gate.evaluate({
            "content": "Some factual content.",
            "url": "https://www.reuters.com/article",
            "query": "test",
        })
        assessment = result["source_assessment"]
        self.assertEqual(assessment["domain"], "reuters.com")
        self.assertEqual(assessment["stance"], "centrist")

    # --- Contradiction detection tests ---

    def test_evaluate_flags_contradiction_for_manual_review(self):
        """Content contradicting existing RL should be flagged for manual review."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "entries": [
                    {
                        "content": (
                            "The vaccine is safe and effective. Clinical trials show it "
                            "is proven to prevent serious illness. Researchers confirm "
                            "it causes no significant side effects."
                        ),
                        "source": "https://www.cdc.gov",
                    }
                ]
            }, f)
            rl_path = f.name

        try:
            result = self.gate.evaluate({
                "content": (
                    "The vaccine is dangerous and ineffective. Studies show it is unproven "
                    "and causes harmful side effects. Experts oppose its use."
                ),
                "url": "https://example.com",
                "query": "vaccine safety",
            }, existing_rl_path=rl_path)

            self.assertEqual(result["status"], "manual_review")
            self.assertGreater(len(result["contradictions"]), 0)
        finally:
            os.unlink(rl_path)

    def test_evaluate_no_contradiction_approves(self):
        """Non-contradictory content should be approved."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "entries": [
                    {
                        "content": (
                            "Python is a high-level programming language created by Guido van Rossum. "
                            "It supports multiple paradigms and has a large standard library."
                        ),
                        "source": "https://python.org",
                    }
                ]
            }, f)
            rl_path = f.name

        try:
            result = self.gate.evaluate({
                "content": (
                    "Python 3.12 introduced the typing module improvements and faster interpreter. "
                    "The release includes performance optimizations."
                ),
                "url": "https://python.org",
                "query": "python 3.12 features",
            }, existing_rl_path=rl_path)

            # Should be approved — no contradiction, low sensitivity
            self.assertEqual(result["status"], "approved")
        finally:
            os.unlink(rl_path)

    def test_evaluate_missing_rl_file_ignores_contradiction_check(self):
        """Missing RL file should not cause failure."""
        result = self.gate.evaluate({
            "content": "Some content.",
            "url": "https://example.com",
            "query": "test",
        }, existing_rl_path="/nonexistent/path.json")

        # Should still return a valid result (not crash)
        self.assertIn("status", result)
        self.assertEqual(result["status"], "approved")

    def test_evaluate_existing_rl_as_list(self):
        """RL file as a list of entries should work."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([
                {"content": "Fact one about python.", "source": "https://python.org"},
            ], f)
            rl_path = f.name

        try:
            result = self.gate.evaluate({
                "content": "Python is a programming language.",
                "url": "https://example.com",
                "query": "python basics",
            }, existing_rl_path=rl_path)

            self.assertIn("status", result)
        finally:
            os.unlink(rl_path)


class TestHelperFunctions(unittest.TestCase):
    """Test internal helper functions."""

    def test_extract_domain_from_url(self):
        self.assertEqual(_extract_domain("https://www.nytimes.com/article"), "nytimes.com")
        self.assertEqual(_extract_domain("http://reuters.com/news"), "reuters.com")
        self.assertIsNone(_extract_domain(""))
        self.assertIsNone(_extract_domain("not a url"))

    def test_get_source_stance_known_domains(self):
        self.assertEqual(_get_source_stance("nytimes.com"), "center-left")
        self.assertEqual(_get_source_stance("wsj.com"), "center-right")
        self.assertEqual(_get_source_stance("reuters.com"), "centrist")

    def test_get_source_stance_unknown_domain(self):
        self.assertEqual(_get_source_stance("unknown-site.xyz"), "unknown")
        self.assertEqual(_get_source_stance(None), "unknown")

    def test_get_source_stance_gov_tld(self):
        self.assertEqual(_get_source_stance("www.cdc.gov"), "institutional")
        self.assertEqual(_get_source_stance("data.nasa.gov"), "institutional")

    def test_get_source_stance_edu_tld(self):
        self.assertEqual(_get_source_stance("mit.edu"), "academic")


class TestConstants(unittest.TestCase):
    """Verify configuration constants are set correctly."""

    def test_default_sensitivity_is_high(self):
        self.assertEqual(DEFAULT_SENSITIVITY, "high")

    def test_bias_confidence_threshold(self):
        self.assertIsInstance(BIAS_CONFIDENCE_THRESHOLD, float)
        self.assertAlmostEqual(BIAS_CONFIDENCE_THRESHOLD, 0.5)

    def test_rl_contradiction_threshold(self):
        self.assertIsInstance(RL_CONTRADICTION_THRESHOLD, int)
        self.assertEqual(RL_CONTRADICTION_THRESHOLD, 3)


class TestGracefulDegradation(unittest.TestCase):
    """Verify graceful degradation — no unhandled exceptions."""

    def test_classifier_never_raises(self):
        classifier = TopicSensitivityClassifier()
        # Should not raise on any input type
        for bad_input in [None, "", 0, [], {}, object(), b"bytes"]:
            try:
                result = classifier.classify(bad_input)  # type: ignore[arg-type]
                self.assertIn(result, ["low", "high"])
            except Exception as e:
                self.fail(f"classify({bad_input!r}) raised {type(e).__name__}: {e}")

    def test_detect_bias_never_raises(self):
        gate = ScrutinyGate()
        for bad_input in [None, "", 0, [], {}, object(), b"bytes"]:
            try:
                result = gate.detect_bias(bad_input)  # type: ignore[arg-type]
                self.assertIn("detected", result)
            except Exception as e:
                self.fail(f"detect_bias({bad_input!r}) raised {type(e).__name__}: {e}")

    def test_vet_results_never_raises(self):
        gate = ScrutinyGate()
        for bad_results in [None, "", 0]:
            try:
                result = gate.vet_results(bad_results, query="test")  # type: ignore[arg-type]
                self.assertIn("vetted_results", result)
            except Exception as e:
                self.fail(f"vet_results({bad_results!r}) raised {type(e).__name__}: {e}")

    def test_rl_gate_never_raises(self):
        gate = RLIngestionGate()
        for bad_data in [None, "", 0, [], {}]:
            try:
                result = gate.evaluate(bad_data)  # type: ignore[arg-type]
                self.assertIn("status", result)
            except Exception as e:
                self.fail(f"evaluate({bad_data!r}) raised {type(e).__name__}: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
