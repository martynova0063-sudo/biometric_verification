---
name: Verification multi-face policy
description: Product behavior for multiple candidate detections during 1:1 face verification.
---

Any sampled frame with more than one detected face candidate must produce a visible red warning in the session transcript, regardless of the overall ratio. The warning threshold can mark elevated prevalence, but the session must not be blocked while the ratio is at or below the configured block threshold. Only a ratio strictly above that threshold blocks reliable evaluation and routes the session to retry/manual review. The default strategy should score candidates against the reference embedding and choose the closest match. Keep `max` and `skip_frame` available as explicit alternatives; `skip_frame` skips only that frame, not the entire session.

**Why:** The user requires a red warning for any multi-face frame, but no refusal unless the configured block threshold is exceeded; detector candidates can also include false detections.

**How to apply:** Treat detector boxes as candidates rather than confirmed people. Apply the warn threshold as informational and block only above the configured block threshold; independent match, liveness, and capture-quality rules still apply.
