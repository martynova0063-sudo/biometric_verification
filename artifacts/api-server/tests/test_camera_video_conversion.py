import shutil
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from app.services import convert_camera_video_to_mp4


FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


@unittest.skipUnless(FFMPEG and FFPROBE, "ffmpeg and ffprobe are required")
class CameraVideoConversionTests(unittest.TestCase):
    def test_square_camera_video_is_saved_as_640_by_480(self):
        with tempfile.TemporaryDirectory(prefix="ukz-camera-test-") as source_dir:
            for extension, codec, encoder_args in (
                ("mp4", "mpeg4", []),
                ("webm", "libvpx", ["-deadline", "realtime", "-cpu-used", "8"]),
            ):
                with self.subTest(extension=extension):
                    source_path = Path(source_dir) / f"camera-source.{extension}"
                    generated = subprocess.run(
                        [
                            FFMPEG,
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-f",
                            "lavfi",
                            "-i",
                            "color=c=black:s=400x400:r=15:d=1",
                            "-an",
                            "-c:v",
                            codec,
                            *encoder_args,
                            "-pix_fmt",
                            "yuv420p",
                            str(source_path),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(generated.returncode, 0, generated.stderr)

                    output_path, output_dir = convert_camera_video_to_mp4(
                        source_path.read_bytes(),
                        max_bytes=10 * 1024 * 1024,
                        source_extension=extension,
                    )
                    try:
                        probed = subprocess.run(
                    [
                        FFPROBE,
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=width,height,avg_frame_rate",
                        "-of",
                        "json",
                        str(output_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                        self.assertEqual(probed.returncode, 0, probed.stderr)
                        stream = json.loads(probed.stdout)["streams"][0]
                        self.assertEqual((stream["width"], stream["height"]), (640, 480))
                        numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
                        self.assertGreaterEqual(numerator / denominator, 15)
                    finally:
                        shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
