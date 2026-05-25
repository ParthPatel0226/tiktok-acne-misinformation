"""
Video Analytics — Extract visual features from video frames
=============================================================
Uses Gemini 2.5 Flash (FREE tier) on extracted video frames to classify:

  VISUAL:   video_shot_complexity, No._people_video, face_visibility
            indoor_outdoor, creator_gender, creator_age_range
            creator_race_visual, video_format, Video_type
  CONTENT:  skin_condition_type, condition_extent, before_after_claim

Requirements:
    pip install google-generativeai Pillow

Usage:
    python video_analytics.py --input TikTok_Acne_Classified.csv --video-dir tiktok_videos/ \
        --output TikTok_Acne_Visual.csv --api-key YOUR_GEMINI_KEY

    # Test on 5 videos
    python video_analytics.py --input data.csv --video-dir vids/ --output test.csv \
        --api-key KEY --limit 5

Rate limits (Gemini free tier): 15 RPM → 4.5s sleep between requests
Total time: ~49 minutes for 731 videos
"""

import csv
import json
import time
import os
import sys
import argparse
import subprocess
from pathlib import Path

try:
    import google.generativeai as genai
    from PIL import Image
except ImportError:
    print("ERROR: Install: pip install google-generativeai Pillow")
    sys.exit(1)


VISION_PROMPT = """Analyze these video frames from a TikTok video about acne/skincare.
Return ONLY a JSON object with these fields (no markdown, no backticks):

{
  "creator_gender": "male|female|unknown",
  "creator_age_range": "under_18|18-25|26-35|36-50|50+|unknown",
  "creator_race_visual": "perceived race/ethnicity or 'unknown'",
  "No_people_video": 1,
  "face_visibility": "full_face|partial_face|no_face|back_of_head",
  "video_shot_complexity": "1=simple selfie | 2=basic setup | 3=professional | 4=cinematic",
  "indoor_outdoor": "indoor|outdoor|mixed",
  "video_format": "selfie|tutorial|talking_head|montage|text_overlay|voiceover|other",
  "Video_type": "self_recorded|ai_generated|professional|other",
  "skin_condition_visible": "yes|no",
  "condition_extent": "mild|moderate|severe|unclear",
  "before_after_visual": "yes_visible|no|unclear"
}
"""


def extract_frames(video_path, output_dir, num_frames=3):
    """Extract frames from video at start, middle, end."""
    os.makedirs(output_dir, exist_ok=True)
    frames = []
    
    # Get video duration
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True
        )
        duration = float(result.stdout.strip())
    except:
        duration = 30  # default
    
    timestamps = [1, duration / 2, max(duration - 1, 2)]
    
    for i, ts in enumerate(timestamps):
        frame_path = os.path.join(output_dir, f"frame_{i}.jpg")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
                 "-vframes", "1", "-q:v", "2", frame_path],
                capture_output=True, timeout=10
            )
            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 1000:
                frames.append(frame_path)
        except:
            pass
    
    return frames


def analyze_frames(model, frame_paths):
    """Send frames to Gemini for analysis."""
    images = []
    for fp in frame_paths:
        try:
            img = Image.open(fp)
            images.append(img)
        except:
            pass
    
    if not images:
        return None
    
    try:
        response = model.generate_content([VISION_PROMPT] + images)
        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return None
    except Exception as e:
        print(f"  Gemini error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Extract visual features from TikTok video frames")
    parser.add_argument("--input", required=True, help="Input CSV")
    parser.add_argument("--video-dir", required=True, help="Directory with downloaded videos")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    # Add visual columns
    visual_cols = ["creator_gender", "creator_age_range", "creator_race_visual",
                   "No_people_video", "face_visibility", "video_shot_complexity",
                   "indoor_outdoor", "video_format", "Video_type",
                   "skin_condition_visible", "condition_extent", "before_after_visual"]
    for col in visual_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    total = min(len(rows), args.limit) if args.limit else len(rows)
    analyzed = 0
    frames_dir = os.path.join(args.video_dir, "_frames")

    print(f"Analyzing {total} videos...")

    for i, row in enumerate(rows):
        if args.limit and i >= args.limit:
            break
        
        if args.resume and row.get("creator_gender", "").strip():
            continue

        video_id = row.get("video_id", f"row_{i}")
        
        # Find video file
        video_path = None
        for ext in ["mp4", "webm", "mkv"]:
            candidate = os.path.join(args.video_dir, f"{video_id}.{ext}")
            if os.path.exists(candidate):
                video_path = candidate
                break
        
        if not video_path:
            print(f"  [{i+1}/{total}] {video_id} — no video file, skipping")
            continue

        print(f"  [{i+1}/{total}] {video_id}...", end=" ", flush=True)

        # Extract frames
        vid_frames_dir = os.path.join(frames_dir, str(video_id))
        frame_paths = extract_frames(video_path, vid_frames_dir)
        
        if not frame_paths:
            print("no frames extracted")
            continue

        # Analyze with Gemini
        result = analyze_frames(model, frame_paths)
        
        if result:
            for key in visual_cols:
                if key in result:
                    row[key] = str(result[key])
            analyzed += 1
            print(f"OK ({len(frame_paths)} frames)")
        else:
            print("analysis failed")

        # Checkpoint
        if (i + 1) % 20 == 0:
            with open(args.output, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

        # Rate limit: 15 RPM = 4s minimum between requests
        time.sleep(4.5)

    # Final save
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Analyzed {analyzed}/{total} videos → {args.output}")


if __name__ == "__main__":
    main()
