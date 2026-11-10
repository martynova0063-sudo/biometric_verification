---
name: Workspace runtime conventions
description: Runtime details learned while wiring the UKZ web and Python services in this monorepo.
---

Package scripts invoked through `pnpm --filter` run with the package directory as their working directory; commands should use paths relative to that directory rather than prefixing `artifacts/<package>`.

**Why:** Prefixing the artifact path caused the first migration command to fail because it attempted to enter the package directory twice.

**How to apply:** Keep filtered package scripts self-contained and relative to the package root; use repository-root paths only in artifact service production commands.