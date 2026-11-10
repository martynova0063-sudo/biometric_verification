---
name: OpenCV wheel compatibility
description: OpenCV runtime compatibility when DeepFace and headless OpenCV are both installed in this Python 3.13 workspace.
---

When DeepFace's `opencv-python` dependency is installed beside the workspace's `opencv-python-headless` wheel, keep their versions aligned. Mismatched wheels can overwrite or remove shared `cv2` files, leaving a namespace module without native APIs such as `setNumThreads`. The standard OpenCV wheel also needs the system libraries `libGL`, `glib`, and `xorg.libxcb` in this workspace.

**Why:** Removing an optional vision package exposed an existing mismatch between OpenCV 5.0 and 4.12 wheels; the backend then failed during import until the wheels were aligned.

**How to apply:** After changing Python vision dependencies, inspect the resolved OpenCV versions and verify both `import cv2` and `cv2.setNumThreads` before restarting the API workflow.
