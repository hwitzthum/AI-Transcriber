import os
import subprocess

def create_test_video(output_path="tests/test_video.mp4"):
    # Generate a 5-second video with audio using ffmpeg
    # Video: red color solid
    # Audio: sine wave beep
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=320x240:d=5",
        "-f", "lavfi", "-i", "sine=f=440:d=5",
        "-c:v", "libx264", "-c:a", "aac", "-shortest",
        output_path
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
    print(f"Created {output_path}")

if __name__ == "__main__":
    create_test_video()
