
import argparse
from pathlib import Path

from faster_whisper import WhisperModel

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".mp4", ".ogg", ".webm", ".opus"}


def find_audio_files(input_path: str):
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [path]
        raise ValueError(f"Unsupported audio format: {path.suffix}")

    if path.is_dir():
        return sorted(
            p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    raise ValueError(f"Unsupported path type: {path}")


def transcribe_audio(audio_path: Path, model: WhisperModel):
    segments, _ = model.transcribe(str(audio_path))
    text = "".join(segment.text for segment in segments).strip()
    return text


def save_transcript(audio_path: Path, text: str, output_dir: Path | None = None) -> Path:
    output_path = (output_dir or audio_path.parent) / f"{audio_path.stem}.txt"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert audio files in a folder to text offline")
    parser.add_argument("path", nargs="?", default=".", help="Audio file or folder to transcribe")
    parser.add_argument("--model", default="tiny", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--compute-type", default="int8", help="Compute type for the model")
    parser.add_argument("--output-dir", default=None, help="Folder to save text files into")
    args = parser.parse_args()

    audio_files = find_audio_files(args.path)
    if not audio_files:
        raise SystemExit(f"No supported audio files were found in {args.path}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    for audio_path in audio_files:
        print(f"Transcribing: {audio_path.name}")
        text = transcribe_audio(audio_path, model)
        output_path = save_transcript(audio_path, text, output_dir)
        print(f"Saved transcript: {output_path}")
        print(text)
        print("-" * 40)


if __name__ == "__main__":
    main()
