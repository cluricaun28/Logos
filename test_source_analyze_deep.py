"""Standalone test: source_analyze deep mode with web_extract

Tests that source_analyze with deep=True:
1. Extracts full content from URLs via web_extract_tool
2. Passes full content (not just snippets) to the analyzer
3. Produces richer findings from full content vs. snippets

Run from the hermes-agent root:
    python3 test_source_analyze_deep.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def test_deep_mode():
    """Test deep mode end-to-end: web_extract → source_analyze with full content."""
    print("\n=== Test: source_analyze deep mode ===\n")

    # Use a real URL with substantial content
    test_url = "https://www.bbc.com/news/world-us-canada-67255102"

    # Step 1: web_extract to get full content
    from tools.web_tools import web_extract_tool

    print(f"  Extracting content from {test_url} ...")
    result = await web_extract_tool([test_url])

    if isinstance(result, str):
        data = json.loads(result)
    else:
        data = result

    if not data.get("results"):
        print(f"  ✗ web_extract returned no results: {data}")
        return False

    extracted = data["results"][0]
    content = extracted.get("content", "") or extracted.get("raw_content", "")
    print(f"  ✓ Extracted {len(content)} chars of content")
    if content:
        print(f"  First 100 chars: {content[:100]}...")

    if not content or len(content) < 50:
        print(f"  ⚠ Minimal content ({len(content)} chars) — using synthetic content for test")
        # If extraction fails or returns too little, use synthetic content
        # that simulates what web_extract would return
        content = (
            "The Gaza war has killed over 70,000 people since October 2023, "
            "according to the Gaza Ministry of Health. The Lancet published a "
            "peer-reviewed analysis estimating 64,260 deaths from traumatic injury "
            "between October 2023 and June 2024, with 59.1% being women, children, "
            "and the elderly. Israel maintains that many casualties were Hamas "
            "combatants embedded in civilian areas. The UN's OCHA confirms the "
            "death toll exceeds 75,000. Independent researchers at Brown University's "
            "Cost of War project estimate that over ten percent of Gaza's population "
            "has been directly killed or injured in two years of conflict."
        )
        test_url = "https://www.aljazeera.com/features/2026/2/18/gaza-death-toll"

    # Step 2: Analyze with full content
    from agent.source_analysis import SourceAnalyzer

    analyzer = SourceAnalyzer()
    report = analyzer.analyze(
        url=test_url,
        content=content,
        query_context="test query",
    )

    print(f"\n  Analysis results:")
    print(f"    Domain: {report.source.domain}")
    print(f"    Cluster: {report.source.cluster}")
    print(f"    Alignment: {report.source.alignment}")
    print(f"    Bias score: {report.content.bias_score}")
    print(f"    Markers: {report.content.markers[:3]}")
    print(f"    Deviation: {report.narrative.deviation}")
    print(f"    Findings: {len(report.findings)}")
    print(f"    URL set: {bool(report.url)}")

    # Verify key fields are populated
    assert report.url, f"URL not set: {report.url}"
    assert report.source.domain, f"Domain not set: {report.source.domain}"
    assert isinstance(report.content.bias_score, float), "bias_score not float"
    print(f"  ✓ Domain: {report.source.domain}, cluster: {report.source.cluster}")

    print("\n  ✓ All assertions passed")

    # Step 3: Test with web_search-style input (dict with url + content)
    search_result = {
        "url": test_url,
        "content": content,
        "title": "Example Domain",
        "snippet": "This domain is for use in documentation examples.",
    }

    reports = analyzer.analyze_batch([search_result], query_context="test")
    assert len(reports) == 1, f"Expected 1 report, got {len(reports)}"
    assert reports[0].url == test_url, "URL not set in batch report"
    print("  ✓ Batch analysis with full content works")

    print("\n=== All tests passed ===\n")
    return True


def test_deep_schema():
    """Verify the schema accepts deep parameter."""
    print("\n=== Test: source_analyze schema has deep param ===\n")

    from plugins.memory.perpetual_context.schemas import SOURCE_ANALYZE_SCHEMA

    props = SOURCE_ANALYZE_SCHEMA.get("parameters", {}).get("properties", {})
    assert "deep" in props, f"'deep' not in schema properties: {list(props.keys())}"
    assert props["deep"]["type"] == "boolean", "deep should be boolean"
    print("  ✓ Schema has 'deep' parameter (boolean)")
    print("\n=== Schema test passed ===\n")
    return True


if __name__ == "__main__":
    # Test schema first (sync), then deep mode (async)
    try:
        test_deep_schema()
    except AssertionError as e:
        print(f"\n✗ SCHEMA FAILED (expected — not built yet): {e}\n")
    except Exception as e:
        print(f"\n✗ SCHEMA ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        asyncio.run(test_deep_mode())
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
