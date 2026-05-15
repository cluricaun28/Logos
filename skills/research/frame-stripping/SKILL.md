---
name: frame-stripping
description: >
  Strip ideological framing from web content and re-present facts through
  the user's worldview baseline. Uses the worldview profile (if configured)
  to guide reframing. Falls back to neutral fact presentation if no profile exists.
---

# Frame Stripping

Strip source framing and re-present factual content through the user's
worldview lens. Works on extracted web content from `web_extract` or
`browser_snapshot`.

## When to Use

- After `web_extract` or browser tools return content from a source with known bias
- When the user asks for an article summarized or explained
- When presenting research results that include ideologically-framed sources

## Core Principle

The user does not need to know where a source stands on an issue. They need
to know what happened, presented in a way they can use without wading through
editorial spin designed to shape their interpretation.

The goal is **framing sovereignty**: the user's frame replaces the author's
frame. The facts remain intact; the interpretive language becomes the user's
own.

## Process

### Step 0: Pre-Stripping Checks

Before frame stripping, run through these checks:

1. **Beneficiary Analysis (Cui Bono):** Who benefits from this outcome?
2. **Premise Detection:** What unexamined premises does this source assume?
3. **Compulsion Detection:** Is the article framing coercion as virtue?
4. **Negative-Signal Detection:** Are aligned sources reporting badly for one side?
5. **Adversarial Cross-Reference:** Do opposing sources report the same event
   differently? The gap is the framing.
6. **Causal Chain Analysis:** Don't stop at "X happened." Trace the cause chain.
7. **Institutional Capture Detection:** Do those who profit from a broken system
   control its "solutions"?
8. **Victim-Direction Framing:** Who does the propagandist want you to sympathize
   with and who do they want you to attack?

### Step 1: Identify Source Framing

If a worldview profile exists (`~/.hermes/skills/worldview-profile/SKILL.md`),
load it for context. Identify:

- Loaded terminology (god terms, devil terms, weasel words)
- Omissions (what the source doesn't cover)
- Causal attribution (who do they blame?)
- Moral framing (who is victim, who is villain?)

### Step 2: Extract Facts from Framing

Separate:
- **Facts:** Verifiable claims (what happened, who said it, dates, numbers)
- **Framing:** Evaluative language, loaded terms, moral judgments, omissions

### Step 3: Reframe Through User's Worldview

Load the worldview profile if it exists. Use the user's terminology and
perspective for disputed concepts. If no profile exists, present facts
neutrally with clear source attribution.

1. Replace loaded terminology with neutral or user-aligned equivalents
2. Attribute clearly: "The [Source] reports that..."
3. Present causal analysis from the user's perspective
4. If uncertain about the user's position, state assumptions clearly

### Step 4: Cross-Reference

Check sources with opposing biases for facts both agree on. That's the
reliable core.

### Step 5: Present

Present the reframed content with source attribution and heuristic insights
where relevant.

## Output Format

`[Source: {domain} — {ideological cluster} — {alignment}]`

Then present reframed facts with interpretation through the user's worldview.

## Example

**Input (ideological source):** "Pro-Choice advocates defend women's fundamental
rights as healthcare policies restrict access."

**Output:**
`[Source: example.com — Progressive — Opposed]`

The source reports that advocates for abortion access are opposing new healthcare
policies that limit abortion services. The article frames this as a defense of
women's rights. [Apply user's worldview profile for reframing terminology.]

## Critical Rules

1. Don't mistake "both sides" hedging for neutrality
2. Follow the money: who funds the source and what are their goals?
3. If the user has a worldview profile, use it. If not, be neutral but
   attribute clearly.
4. When uncertain, state assumptions.
