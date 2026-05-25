"""
TikTok Video Downloader + Whisper Transcription Pipeline
---------------------------------------------------------
Downloads all TikTok videos and transcribes audio using Whisper.

Requirements (run once):
    pip install yt-dlp openai-whisper

    # On Windows also install ffmpeg:
    # Download from https://www.gyan.dev/ffmpeg/builds/
    # Add to PATH, or place ffmpeg.exe in same folder as this script

Usage:
    # Full run
    python download_and_transcribe.py --input TikTok_Acne_-_Merged_Data.csv --output TikTok_Acne_Transcribed.csv --cookies cookies.txt

    # Test on 5 videos first
    python download_and_transcribe.py --input TikTok_Acne_-_Merged_Data.csv --output test_transcribed.csv --cookies cookies.txt --limit 5

    # Use faster/smaller whisper model
    python download_and_transcribe.py --input data.csv --output out.csv --cookies cookies.txt --model base

    # Resume interrupted run (skips already-transcribed rows)
    python download_and_transcribe.py --input data.csv --output out.csv --cookies cookies.txt --resume

Whisper model sizes:
    tiny   - fastest, least accurate, ~1GB RAM
    base   - fast, decent accuracy, ~1GB RAM        <- recommended for bulk
    small  - balanced, ~2GB RAM
    medium - accurate, ~5GB RAM
    large-v3 - most accurate, ~10GB RAM, slow       <- use if you have GPU

Estimated time (base model, CPU):
    ~5-10 seconds per video
    731 videos = ~1-2 hours total
"""

import csv
import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
from pathlib import Path


def check_dependencies():
    """Check all required dependencies are installed."""
    missing = []
    try:
        import yt_dlp
    except ImportError:
        missing.append("yt-dlp (run: pip install yt-dlp)")
    try:
        import whisper
    except ImportError:
        missing.append("openai-whisper (run: pip install openai-whisper)")
    
    # Check ffmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
        if result.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError:
        missing.append("ffmpeg — download from https://www.gyan.dev/ffmpeg/builds/ and add to PATH")
    
    if missing:
        print("ERROR: Missing dependencies:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)
    print("All dependencies found.")


def download_audio(video_url, output_dir, cookies_file=None):
    """Download audio from TikTok video using yt-dlp."""
    import yt_dlp

    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
    
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 3,
    }
    
    if cookies_file and os.path.exists(cookies_file):
        ydl_opts["cookiefile"] = cookies_file

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            video_id = info.get("id", "unknown")
            audio_path = os.path.join(output_dir, f"{video_id}.mp3")
            if os.path.exists(audio_path):
                return audio_path, video_id
            # Try other extensions
            for ext in ["m4a", "webm", "ogg", "wav"]:
                alt_path = os.path.join(output_dir, f"{video_id}.{ext}")
                if os.path.exists(alt_path):
                    return alt_path, video_id
            return None, video_id
    except Exception as e:
        return None, str(e)


def transcribe_audio(audio_path, model):
    """Transcribe audio file using Whisper."""
    try:
        result = model.transcribe(audio_path, language=None)
        return result.get("text", "").strip(), result.get("language", "unknown")
    except Exception as e:
        return "", f"error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Download TikTok videos and transcribe with Whisper")
    parser.add_argument("--input", required=True, help="Input CSV file with video_url column")
    parser.add_argument("--output", required=True, help="Output CSV file with transcript column")
    parser.add_argument("--cookies", default=None, help="Path to cookies.txt file from TikTok")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--limit", type=int, default=None, help="Process only N videos (for testing)")
    parser.add_argument("--resume", action="store_true", help="Skip already-transcribed rows")
    parser.add_argument("--keep-audio", action="store_true", help="Keep audio files after transcription")
    parser.add_argument("--audio-dir", default="tiktok_audio", help="Directory for audio files")
    args = parser.parse_args()

    check_dependencies()
    import whisper

    # Create audio directory
    os.makedirs(args.audio_dir, exist_ok=True)

    # Load Whisper model
    print(f"Loading Whisper {args.model} model...")
    whisper_model = whisper.load_model(args.model)
    print("Model loaded.")

    # Read input CSV
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    # Add transcript columns if missing
    for col in ["transcript", "audio_language"]:
        if col not in fieldnames:
            fieldnames.append(col)

    total = min(len(rows), args.limit) if args.limit else len(rows)
    downloaded = 0
    transcribed = 0
    unavailable = 0
    failed = 0

    # Create mapping file
    mapping_path = os.path.join(args.audio_dir, "video_id_mapping.csv")
    
    print(f"\nProcessing {total} videos...")
    print(f"Audio directory: {args.audio_dir}")
    print(f"Cookies: {'yes' if args.cookies else 'no'}")
    print("=" * 55)

    for i, row in enumerate(rows):
        if args.limit and i >= args.limit:
            break

        video_url = row.get("video_url", "").strip()
        video_id = row.get("video_id", f"row_{i}")

        # Skip if already transcribed and resuming
        if args.resume and row.get("transcript", "").strip():
            continue

        if not video_url:
            row["transcript"] = ""
            row["audio_language"] = ""
            unavailable += 1
            continue

        print(f"  [{i+1}/{total}] {video_id}...", end=" ", flush=True)

        # Download
        audio_path, dl_info = download_audio(video_url, args.audio_dir, args.cookies)
        
        if audio_path and os.path.exists(audio_path):
            downloaded += 1
            
            # Transcribe
            transcript, language = transcribe_audio(audio_path, whisper_model)
            row["transcript"] = transcript
            row["audio_language"] = language
            
            if transcript:
                transcribed += 1
                print(f"OK ({len(transcript)} chars, {language})")
            else:
                print("no speech detected")

            # Clean up audio
            if not args.keep_audio:
                try:
                    os.remove(audio_path)
                except:
                    pass
        else:
            row["transcript"] = ""
            row["audio_language"] = ""
            
            if "unavailable" in str(dl_info).lower() or "private" in str(dl_info).lower():
                unavailable += 1
                print("unavailable")
            else:
                failed += 1
                print(f"failed: {str(dl_info)[:50]}")

        # Save checkpoint every 10 videos
        if (i + 1) % 10 == 0:
            with open(args.output, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            print(f"  [Checkpoint saved: {i+1}/{total}]")

        # Small delay between downloads
        time.sleep(1)

    # Final save
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*55}")
    print(f"COMPLETE")
    print(f"  Downloaded:   {downloaded}")
    print(f"  Transcribed:  {transcribed}")
    print(f"  Unavailable:  {unavailable}")
    print(f"  Failed:       {failed}")
    print(f"  Output:       {args.output}")
    print(f"{'='*55}")
    print("Next step: run classify_videos_v2.py on this output file.")
    print("The classifier will use transcripts for much better accuracy.")


if __name__ == "__main__":
    main()
