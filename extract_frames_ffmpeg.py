import os, subprocess

INPUT_DIR = r"C:/Users/serah/Videos/basket3"
OUTPUT_DIR = r"C:/Users/serah/Videos/basket_frames"
FPS = 2

FFMPEG = "ffmpeg"

# SDR 영상 톤업용 필터 (밝기 + 감마 + 대비)
FILTER = (
    f"fps={FPS},"
    "eq=brightness=0.05:contrast=1.15:saturation=1.1:gamma=1.05,"
    "scale=-1:1080,format=yuv420p"
)

def extract_frames():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith((".mov", ".mp4")):
            continue
        name = os.path.splitext(fname)[0]
        in_path = os.path.join(INPUT_DIR, fname)
        out_dir = os.path.join(OUTPUT_DIR, name)
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            FFMPEG, "-i", in_path,
            "-vf", FILTER,
            os.path.join(out_dir, "%04d.jpg"),
            "-hide_banner", "-loglevel", "error", "-y"
        ]
        subprocess.run(cmd, check=True)
        print(f"[OK] {fname} → {out_dir}")

if __name__ == "__main__":
    extract_frames()
