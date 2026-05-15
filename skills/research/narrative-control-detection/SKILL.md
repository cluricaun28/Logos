---
name: narrative-control-detection
description: >
  Detect coordinated narrative manipulation in media coverage. Identifies
  when stories are being framed, reframed, or suppressed through information
  warfare techniques.
---

# Narrative Control Detection

Detect when media organizations are coordinating to suppress, distort, or
amplify specific narratives through information warfare techniques.

## Core Principle

Media organizations are actors with specific interests, funders, and
worldviews. When a story threatens those interests, they suppress it.
When it serves those interests, they amplify it. The pattern is
predictable.

## Information Warfare Pattern (Six Phases)

### Phase 1: Initial Break
Someone breaks the real story (fringe outlet, social media, mistake).

**Signal:** The first article contains details that later disappear.

### Phase 2: Narrative Assessment
The media ecosystem evaluates the story against its interests.

### Phase 3: Narrative Shift
Follow-up articles change the frame:
- De-emphasize threatening details
- Re-emphasize acceptable framing
- Introduce counter-narratives
- Depersonalize the subject

**Signal:** The second wave omits details from the first.

### Phase 4: Article Removal/Rewriting
The original article is pulled, rewritten, or buried behind paywalls.

**Signal:** URL returns 404 or different content.

### Phase 5: Flood the Zone
The sanitized narrative floods search results.

**Signal:** Top results all tell the same version.

### Phase 6: Narrative Entrenchment
The sanitized version becomes "the story."

**Signal:** Later coverage refers only to the sanitized version.

## Detection Method

### Step 1: Map the Initial Break
Search for early articles, social media posts, and fringe outlet coverage.

### Step 2: Map the Narrative Shift
Check if early articles were removed or rewritten. Compare content changes.

### Step 3: Map the Flood
Check social media and alternative platforms for details absent from
mainstream coverage.

## Analysis Framework

1. **Who benefits from this narrative?** Follow money and ideology.
2. **What is the source's worldview?** Consult source dossiers.
3. **Is the omission patterned?** A pattern across sources is coordination.
4. **What would a neutral story look like?** Strip all framing; compare.

## Known Techniques

1. **Astroturfing** — fake grassroots movements
2. **Algorithmic manipulation** — micro-targeted content at scale
3. **Narrative laundering** — political agendas framed as moral causes
4. **Coalition management** — stories threatening alliances get suppressed
5. **Flood the zone** — drown dissent in repetition
6. **Platform capture** — moderation aligned with geopolitical agenda
7. **Foreign agent operations** — follow FARA filings

## Output Format

```
[Incident: Brief description]

**Initial break (Date):**
- Source: [outlet] — [key details present]

**Narrative shift (Date range):**
- Omitted details: [what disappeared]
- Added framing: [what was introduced]
- Articles removed/rewritten: [URLs]

**Current narrative (Date):**
- Top results all say: [sanitized version]
- Details absent: [what's missing]

**Who benefits from suppression:**
- [Funders/ideology that benefits]

**Signal assessment:** [High/Medium/Low]

**Facts stripped of framing:**
- [Bulleted verifiable facts]
```
