"""Standalone test: auto-create dossier + update domain-index.json

Run from the hermes-agent root:
    python3 test_dossier_auto_create.py

Tests:
1. Create a new dossier for a domain that doesn't exist
2. Verify the file is created with the template
3. Verify domain-index.json is updated
4. Verify _DossierLookup.lookup() finds the new dossier
5. Clean up the test artifacts
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

# Add agent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from agent.source_analysis import (
    SourceAnalyzer,
    SourceProfile,
    ContentAnalysis,
    NarrativeSignal,
    NewFinding,
    AnalysisReport,
    _RLWriter,
    _DossierLookup,
)

RL_PATH = Path.home() / ".hermes" / "reference-library"
ENTITIES_DIR = RL_PATH / "entities"
INDEX_PATH = ENTITIES_DIR / "domain-index.json"

# Use a test domain that doesn't exist yet
TEST_DOMAIN = "test-source-auto-create.example.com"
TEST_FILE = "test-source-auto-create-example-com-v1.md"  # dots → hyphens
TEST_PATH = ENTITIES_DIR / TEST_FILE


def cleanup():
    """Remove test artifacts."""
    if TEST_PATH.exists():
        TEST_PATH.unlink()
        print(f"  ✓ Cleaned up {TEST_PATH}")

    # Remove from index
    if INDEX_PATH.exists():
        with open(INDEX_PATH) as f:
            data = json.load(f)
        if TEST_DOMAIN in data.get("domains", {}):
            del data["domains"][TEST_DOMAIN]
            with open(INDEX_PATH, "w") as f:
                json.dump(data, f, indent=4)
            print(f"  ✓ Removed {TEST_DOMAIN} from domain-index.json")


def test_auto_create():
    print(f"\n=== Test: auto-create dossier for {TEST_DOMAIN} ===\n")

    # Ensure clean state
    cleanup()

    # Verify it doesn't exist yet
    assert not TEST_PATH.exists(), f"Test file already exists: {TEST_PATH}"
    with open(INDEX_PATH) as f:
        data = json.load(f)
    assert TEST_DOMAIN not in data.get("domains", {}), "Domain already in index"
    print("  ✓ Clean state confirmed")

    # Create findings for the new domain
    findings = [
        NewFinding(
            domain=TEST_DOMAIN,
            category="omits",
            entry="Political funding sources",
            evidence="Query: test query about funding",
        )
    ]

    # Write via _RLWriter
    writer = _RLWriter(RL_PATH)
    result = writer.write(findings)

    # Verify dossier was created
    assert TEST_PATH.exists(), f"Dossier was not created: {TEST_PATH}"
    print(f"  ✓ Dossier created: {TEST_PATH}")

    # Verify content contains expected sections
    content = TEST_PATH.read_text()
    assert TEST_DOMAIN in content, "Domain not in dossier content"
    assert "unknown" in content.lower() or "needs research" in content.lower(), \
        "Dossier should admit unknowns"
    assert "Political funding sources" in content, "Finding not appended"
    print("  ✓ Dossier content verified")

    # Verify domain-index.json updated
    with open(INDEX_PATH) as f:
        data = json.load(f)
    assert TEST_DOMAIN in data.get("domains", {}), "Domain not added to index"
    idx_entry = data["domains"][TEST_DOMAIN]
    assert idx_entry.get("file") == TEST_FILE, f"Index file mismatch: {idx_entry.get('file')}"
    print(f"  ✓ domain-index.json updated: {idx_entry}")

    # Verify _DossierLookup.lookup() finds it
    lookup = _DossierLookup(RL_PATH)
    lookup.ensure_loaded()
    profile = lookup.lookup(f"https://{TEST_DOMAIN}/article")
    assert profile is not None, "Lookup failed to find new dossier"
    assert profile.domain == TEST_DOMAIN, f"Domain mismatch: {profile.domain}"
    print(f"  ✓ _DossierLookup.lookup() finds new dossier: {profile.domain}")

    # Verify the finding is parseable from the dossier
    assert any("Political funding sources" in o for o in profile.omits), \
        f"Finding not in profile.omits: {profile.omits}"
    print("  ✓ Finding parsed back from dossier via lookup")

    # Clean up
    cleanup()

    print("\n=== All tests passed ===\n")
    return True


if __name__ == "__main__":
    try:
        test_auto_create()
    except AssertionError as e:
        print(f"\n✗ FAILED: {e}\n")
        print("Cleanup ...")
        try:
            cleanup()
        except Exception:
            pass
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        try:
            cleanup()
        except Exception:
            pass
        sys.exit(1)
