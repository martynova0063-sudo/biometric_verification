---
name: Not-evaluable verification outcome
description: Product meaning and review routing for sessions that cannot be evaluated reliably.
---

`not_evaluable` is reserved for technical inability to process the inputs, such as unreadable or corrupted media, interrupted transfer/connectivity, or an unavailable required processing component. If readable media fails face-visibility, resolution, duration, detection-count, or multi-face quality checks, use `insufficient_quality`; preserve scores the pipeline actually computed and leave only unavailable values null.

**Why:** The user clarified that `not_evaluable` must indicate a technical processing failure, not a readable recording that fails a quality threshold; employee 1500 was decoded and scored before its values were cleared.

**How to apply:** Use `not_evaluable` for unreadable/damaged uploads, interrupted transfer, or unavailable required components. Route parsed but low-quality content to `insufficient_quality`, keep any computed match/liveness values, and offer retry or manual review. Do not count either status as a failed identity match.
