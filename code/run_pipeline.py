"""
Master Pipeline Runner
======================
Runs the full data enrichment pipeline in the correct order:

  Stage 1: Download audio + transcribe with Whisper
  Stage 2: Scrape comments via Apify  
  Stage 3: NLP classify everything with Claude API

Usage:
    # Run everything
    python run_pipeline.py --input TikTok_Acne_-_Merged_Data.csv --apify-key YOUR_TOKEN --claude-key sk-ant-...

    # Run only specific stages
    python run_pipeline.py --input TikTok_Acne_-_Merged_Data.csv --apify-key TOKEN --claude-key KEY --stages transcribe comments

    # Test mode (5 videos only)
    python run_pipeline.py --input TikTok_Acne_-_Merged_Data.csv --apify-key TOKEN --claude-key KEY --limit 5
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_stage(script, args_list, stage_name):
    print(f"\n{'='*60}")
    print(f"STAGE: {stage_name}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, script] + args_list
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\nERROR: {stage_name} failed with exit code {result.returncode}")
        return False
    
    print(f"\n✓ {stage_name} complete")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run full TikTok acne data pipeline")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--apify-key", required=True, help="Apify API token")
    parser.add_argument("--claude-key", required=True, help="Anthropic API key")
    parser.add_argument("--cookies", default="cookies.txt", help="TikTok cookies file")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size")
    parser.add_argument("--limit", type=int, default=None, help="Process only N videos")
    parser.add_argument("--stages", nargs="+", default=["transcribe", "comments", "classify"],
                        choices=["transcribe", "comments", "classify"],
                        help="Which stages to run")
    args = parser.parse_args()

    input_file = args.input
    base_name = Path(input_file).stem

    # Stage 1: Download + Transcribe
    transcribed_file = f"{base_name}_transcribed.csv"
    if "transcribe" in args.stages:
        stage_args = [
            "--input", input_file,
            "--output", transcribed_file,
            "--model", args.whisper_model,
        ]
        if args.cookies:
            stage_args += ["--cookies", args.cookies]
        if args.limit:
            stage_args += ["--limit", str(args.limit)]
        
        if not run_stage("download_and_transcribe.py", stage_args, "Download + Transcribe"):
            print("Pipeline stopped at Stage 1")
            return
        input_file = transcribed_file

    # Stage 2: Scrape Comments
    comments_file = f"{base_name}_with_comments.csv"
    if "comments" in args.stages:
        stage_args = [
            "--input", input_file,
            "--api-key", args.apify_key,
            "--output-main", comments_file,
            "--output-comments", "tiktok_comments.csv",
        ]
        if args.limit:
            stage_args += ["--limit", str(args.limit)]
        
        if not run_stage("scrape_comments.py", stage_args, "Scrape Comments"):
            print("Pipeline stopped at Stage 2")
            return
        input_file = comments_file

    # Stage 3: NLP Classification
    classified_file = f"{base_name}_classified.csv"
    if "classify" in args.stages:
        stage_args = [
            "--input", input_file,
            "--output", classified_file,
            "--api-key", args.claude_key,
        ]
        if args.limit:
            stage_args += ["--limit", str(args.limit)]
        
        if not run_stage("classify_videos_v2.py", stage_args, "NLP Classification"):
            print("Pipeline stopped at Stage 3")
            return

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"  Final output: {classified_file}")
    print(f"  Comments:     tiktok_comments.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
