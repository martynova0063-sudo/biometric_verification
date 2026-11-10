---
name: Embedding comparison retention
description: Privacy boundary for comparing fresh video embeddings against reference vectors.
---

Generate ArcFace vectors from selected video frames during verification, compare them with the employee's reference vectors, then discard the frame vectors. Persist only aggregate means, variances, outlier counts, and histogram bins in session details; never expose or store the raw video vectors for this feature.

**Why:** Verification videos are intentionally not retained. Aggregate comparison supports quality analysis without creating a durable copy of sensitive biometric vectors; older sessions cannot be backfilled unless a new source video is supplied.

**How to apply:** Keep comparison in the video-processing request path before temporary media is deleted. The user explicitly authorized an opt-in exception: after a successful verification, selected video vectors may be stored only in the separate FAISS employee search index, not on the session record. Keep this opt-in explicit; historical sessions still cannot be backfilled unless a new source video is supplied. For optional LLM explanations, send aggregate metrics only—never raw vectors, video, employee identifiers, or session IDs—and treat the output as technical guidance, not an identity decision.
