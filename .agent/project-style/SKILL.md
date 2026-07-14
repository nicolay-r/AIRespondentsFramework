---
name: project-style
description: >-
  Python coding style for this repository. Use when writing, editing, or reviewing
  Python code in Team-5-Solution. Do not add `from __future__ import annotations`;
  target Python 3.10.
---

# Project Style

## Python version

Target **Python 3.10**.

## Imports

let's don't use future import

Do **not** add:

```python
from __future__ import annotations
```

to new or edited files in this project.

## Typing without `__future__` annotations

Write types that work under Python 3.10 without postponed annotation evaluation:

- Use built-in unions where supported: `str | None`, `tuple[str, ...]`
- Quote forward references when needed: `def foo() -> "PipelineItem":`
- Prefer `from typing import Literal` for literals and other typing helpers when useful

## When editing existing files

- Do not add `from __future__ import annotations` while making other changes
- If a file already has it, leave it unless the user asks to remove it project-wide
- Match surrounding import order and module style in the file you are editing
