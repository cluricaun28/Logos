# Tool System Documentation

## Purpose

This directory contains documentation for every tool available to the Hermes agent. Each tool gets its own page with the full JSON schema, parameters, usage examples, and edge cases.

## Why Document Tools Here?

The system prompt uses **deferred tool injection** — only essential tools (file ops, terminal, web search) have schemas in-context. All other tools are listed by name with a one-line description. Before using a deferred tool, the agent reads its full schema from this directory:

```python
read_file("~/.hermes/reference-library/tools/{tool_name}.md")
```

This keeps the system prompt lean while giving the agent access to complete tool documentation when needed.

## Directory Structure

```
tools/
├── tool-system.md       ← You are here — master index
├── browser/             ← Browser automation tools
│   ├── browser-suite.md  ← Category summary
│   ├── browser_navigate.md
│   ├── browser_snapshot.md
│   └── ...
├── communication/       ← Messaging, TTS, vision
│   ├── send_message.md
│   └── text_to_speech.md
└── process/             ← Background process management
    └── process.md
```

## Creating Tool Pages

Use this template:

```markdown
---
type: tool
name: "tool_name"
category: browser|communication|process|other
created: YYYY-MM-DD
last_updated: YYYY-MM-DD
---

# tool_name

**Category:** category_name
**Description:** One-line description of what the tool does.

## Parameters

```json
{
  "parameter_name": {
    "type": "string|int|bool|array",
    "description": "What this parameter does",
    "required": true|false,
    "default": "default_value"
  }
}
```

## Usage Example

```python
tool_call("tool_name", {"param1": "value1", "param2": "value2"})
```

## Edge Cases and Pitfalls

- Document known failure modes
- List environment variables required
- Note rate limits or timeouts

## Related Tools

- [related_tool](path/to/related.md)
```

## When to Create Tool Pages

Create a tool page when:
1. You encounter a deferred tool for the first time — read its schema, document it here
2. A tool fails repeatedly — add edge cases and troubleshooting notes
3. You discover environment variable requirements — document them permanently
