"""Manual fixture regenerator for ``tests/test_video.mp4``.

This is NOT a pytest test — the leading underscore keeps pytest's
collector from picking it up. It exists so the committed video fixture
(used by ``test_video_audio_extraction``) can be rebuilt if it's ever
deleted or corrupted. Run with::

    uv run python tests/_make_test_video.py

Produces a 5-second 320x240 red video with a 440 Hz sine-wave audio
track, encoded as H.264 + AAC.
"""

import subprocess


def create_test_video(output_path="tests/test_video.mp4"):
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=320x240:d=5",
        "-f", "lavfi", "-i", "sine=f=440:d=5",
        "-c:v", "libx264", "-c:a", "aac", "-shortest",
        output_path,
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
    print(f"Created {output_path}")


if __name__ == "__main__":
    create_test_video()