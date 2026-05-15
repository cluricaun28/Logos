---
name: web-source-bias-research
description: >
  Deep investigative research on news sources. Builds entity dossiers
  tracking ownership, funding, ideological alignment, known omissions,
  and behavioral patterns.
---

# Web Source Bias Research

Build or expand source intelligence dossiers for media organizations,
think tanks, NGOs, and other information producers.

## When to Use

- When a source appears repeatedly in research and you need to understand its bias
- When building out the source intelligence registry
- When a user asks "what do I need to know about this source?"

## Process

### Step 1: Gather Basic Facts

Using the three-tier web pipeline:
1. **web_search** — find the organization, its website, founding date, mission
2. **web_extract** — pull from the organization's "About Us," board pages, annual reports
3. **browser** — for paywalled or dynamic content

### Step 2: Map Ownership and Funding

- Who owns the organization?
- Who funds it? (grants, donors, advertisers, government contracts)
- Follow the money 2-3 levels deep (not just "who donates" but "who funds
  the foundation that donates")
- Check FARA filings for foreign agent connections (US)
- Check board members for corporate board cross-membership

### Step 3: Identify Ideological Alignment

- What is the organization's stated mission and worldview?
- What policy positions do they consistently advocate?
- Which political coalitions do they align with?
- Check their editorial board, key staff, and published positions

### Step 4: Document Known Patterns

- What do they consistently emphasize?
- What do they consistently omit?
- Do they double-standard (apply different rules to ideologically aligned vs
  opposed subjects)?
- Have they been caught in errors or retractions?

### Step 5: Create or Update the Dossier

Save as `~/.hermes/reference-library/organizations/{domain}-v1.md`:

```
---
name: {source-name}
type: source
cluster: {ideological cluster}
alignment: {Aligned/Opposed/Neutral}
---

# {Source Name} ({domain})

## Overview
[What they are, founding, mission]

## Ownership and Funding
[Who owns them, who funds them, key financial relationships]

## Ideological Alignment
[Political/ideological cluster, key positions]

## Known Patterns
### Emphasizes
[What they consistently highlight]

### Omits
[What they consistently ignore]

### Double Standards
[Asymmetric treatment of similar events]

### Notable Deviations
[Times they broke their own pattern — interesting data points]

## Related Sources
[Sources in the same coalition or funding network]

## Assessment
[Overall reliability assessment for different types of queries]
```

## Source Intelligence Registry

The collection of source dossiers in `reference-library/organizations/`
serves as the source intelligence registry. The `domain-index.json` file
provides quick lookup by domain.

## Critical Rules

1. **Document evidence, not opinions.** Every claim in a dossier should be
   traceable to a verifiable source.
2. **Update on new data.** When a source breaks its own pattern, note it.
3. **Distinguish alignment from bias characterization.** "Aligned" means
   agrees with the user's worldview. Bias analysis is separate.
