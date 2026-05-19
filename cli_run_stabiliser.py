import argparse
import sys
from pathlib import Path
from video_stabilizer.app.stabilizer_app import StabilizerApplication
from video_stabilizer.cv2_utils.utils import video_extensions


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-videos", type=str, required=True, help="Folders of videos to stabilize.")
    return parser.parse_args(args)


def get_video_files(folder: Path) -> list[Path]:
    video_files = []
    for file in folder.iterdir():
        if not file.is_file() or file.suffix not in video_extensions:
            continue
        video_files.append(file)
    return video_files


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    videos = get_video_files(Path(args.videos))
    print(f"Found {len(videos)} videos")

    for index, file_path in enumerate(videos):
        print(f"[{index}] {file_path}")
        stb = StabilizerApplication(file_path)
        stb.stabilize()
        stb.close()
