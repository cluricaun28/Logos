---
name: worldview-profile-builder
description: >
  Guide a new user through defining their worldview baseline so the agent can
  do frame-stripping, source analysis, and motive mapping aligned with
  their values rather than generic "neutrality." Produces a configuration
  the agent uses for all future research tasks.
category: research
---

# Worldview Profile Builder

Build a worldview-aligned research profile for a new user. This is the
foundation for frame-stripping, source intelligence, and sovereign
analysis — but the positions come from the *user*, not from the code.

## Trigger

Run this once when a new user first sets up the system, or when they ask
the agent to "learn how I see things" or "calibrate your research to my
viewpoint."

## Process

### Step 1: Interview the User

Ask the user, one at a time, for their positions on the following categories.
Do not present all at once — go one section, get their answer, move on.

**Foundational beliefs:**
- What metaphysical or moral framework do you reason from? (e.g., traditional
  religion, secular humanism, Stoic philosophy, etc.)
- What do you consider non-negotiable truths?

**Political and economic views:**
- How do you view government's role and limits?
- How do you approach foreign policy?
- What are your views on immigration, sovereignty, and national identity?

**Cultural and social views:**
- What are your positions on family, gender, reproductive issues?
- How do you view progressive cultural movements?

**Source preferences:**
- Which sources do you trust and why?
- Which sources do you distrust and why?
- Are there ideological clusters you want to watch for?

**Reasoning style:**
- Do you prefer direct answers or exploratory analysis?
- How should the agent handle disagreement with your positions? (challenge,
  defer, explain the gap)

### Step 2: Generate the Profile

Save the results as `${LOGOS_HOME:-${HERMES_HOME:-$HOME/.logos}}/skills/worldview-profile/SKILL.md` with this
structure:

```
---
name: worldview-profile
description: User-specific worldview baseline for frame-stripping and analysis
---

# User Worldview Profile

## Foundational Framework
[User's metaphysical/moral baseline]

## Political and Economic Positions
[Key positions]

## Cultural and Social Positions
[Key positions]

## Source Intelligence
### Trusted Sources
[Sources the user trusts, with reasons]

### Watch Sources
[Suspicious or biased sources, with known patterns]

### Ideological Markers
[Shibboleths or language patterns the user uses to identify framing]

## Research Instructions
### Frame Stripping
[How to reframe content through the user's lens]

### Motive Analysis
[How to evaluate actor motives]

### Guardrails
[Rules for analysis that reflect the user's worldview]
```

### Step 3: Wire It In

Update the SOUL.md or equivalent system prompt to include:
- "Before presenting news/current events, apply frame-stripping using the
  worldview profile."
- "When researching politically-charged topics, use the user's source
  preferences and motive analysis criteria."

## What This Is Not

This does not impose any worldview. It captures the *user's* worldview and
makes it actionable. The methodology is generic; the content is personal.

The same process works for a progressive secular humanist, a traditional
Christian, a libertarian, or anyone else. The interview extracts their
positions; the profile makes them machine-actionable.
