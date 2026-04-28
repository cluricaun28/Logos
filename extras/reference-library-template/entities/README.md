# Entity Registry — How to Build Entity Pages

## Purpose

The entity registry tracks people, organizations, publications, and institutions with structured metadata about their behavior patterns, funding sources, institutional ties, and credibility on specific topics. This is used to filter source selection before reference library ingestion or presentation.

## Creating Entity Pages

Use this frontmatter format:

```yaml
---
type: entity
name: "Entity Name"
category: person|organization|publication|institutional_group
ownership: ["Owner 1", "Owner 2"]
funders: ["Funder 1", "Funder 2"]
institutional_ties: ["Tie 1", "Tie 2"]
first_seen: YYYY-MM-DD
last_updated: YYYY-MM-DD
credibility_scores:
  topic_area_1: high|medium|low
  topic_area_2: high|medium|low
bias_flags: ["flag1", "flag2"]
---
```

## Example Entity Page

```markdown
---
type: entity
name: "Example Publication"
category: publication
ownership: "Owner name"
funders: ["Subscription revenue", "Advertising"]
institutional_ties: ["Parent company"]
first_seen: YYYY-MM-DD
last_updated: YYYY-MM-DD
credibility_scores:
  technology: medium
  politics: low
bias_flags: ["establishment framing", "corporate advertising dependency"]
---

# Example Publication

## Overview
Brief description of the entity, their mission statement, and public positioning.

## Historical Behavior
Documented patterns of behavior, editorial decisions, or coverage choices that reveal underlying priorities.

## Funding Analysis
Who pays for this entity? Follow the money — funding sources often explain editorial direction.

## Topic-Specific Credibility
This entity is credible on X but not on Y. Be specific about where they have expertise vs. where they have bias.

## References
Links to primary sources, investigations, or documented examples of their behavior patterns.
```

## When to Create Entity Pages

Create an entity page when:
1. You encounter a source repeatedly in research — document its credibility profile
2. A publication has a notable funding conflict — track it for future reference
3. An organization's behavior pattern becomes clear over time — capture the trajectory
4. You need to evaluate source credibility before ingesting information into the RL

## Naming Conventions

- Use kebab-case for filenames: `new-york-times.md`, `daily-wire.md`
- Use full proper names in frontmatter and headings
- Group related entities under subdirectories if needed (e.g., `media/`, `tech-companies/`)
