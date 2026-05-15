---
name: epistemic-framework-design
description: >
  Methodology for designing a source hierarchy and truth pipeline that
  prioritizes the user's trusted sources over web consensus.
---

# Epistemic Framework Design

Design the technical architecture for a truth pipeline that serves the
user's epistemic preferences, not generic "neutrality."

## Trigger Conditions

- User wants to prioritize local/curated knowledge over web search
- User expresses distrust in "consensus" or "authoritative" sources
- User wants to implement a source hierarchy based on trust

## Architecture

### Layer 1: Adaptive Hybrid Weighting

Replace static search weights with dynamic ones based on query type:
- **Bedrock queries (truth/logic):** High weight on curated knowledge,
  low weight on web consensus
- **Exploratory queries (conceptual):** High weight on semantic search
- **Volatile queries (facts/news):** Balanced with recency bias

### Layer 2: Source Intelligence Registry

Build a registry of sources tracking:
- Linguistic markers (shibboleths that cluster sources by ideology)
- Motive mapping (ownership, funding, historical loyalty patterns)
- Integrity signals (do they tell truth at a cost?)

### Layer 3: Signal vs. Noise Filtering

Assess whether source output is information or ideological reassurance:
- **Pep rally detection:** If the article exists primarily to reassure
  its audience, flag as noise
- **Test:** Would a reasonable person make a different decision after
  reading this? If not, it's noise
- **Highest signal:** Sources that contradict the user's baseline but
  present verifiable facts

### Layer 4: Output Synthesis

Present findings as analysis, not summary:
- **Raw signal:** Data without framing
- **Framing analysis:** Which ideological cluster presents the data and why
- **Motive warning:** If the source has a documented interest in the outcome

## Implementation

This architecture is partially implemented in the SourceAnalyzer module
(`agent/source_analysis.py`) and the `source_analyze` tool.

## Pitfalls

- Don't mistake "both sides" hedging for neutrality
- The motives-vs-behavior test is essential: if stated motives and
  actual behavior diverge, the stated motives are likely a cover
- All profiles and marker lists must live in the local Reference Library,
  not in the model's training data
- Maintain the ability to learn from sources you disagree with —
  that's where genuine correction happens
