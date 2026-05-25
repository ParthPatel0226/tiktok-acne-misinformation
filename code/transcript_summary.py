"""
Transcript Summarization
=========================
Summarizes Whisper transcripts into 1-2 sentences using Claude API.
Focuses on: what treatment/product, what claim, what outcome.

Usage:
    python transcript_summary.py --input TikTok_Acne_Transcribed.csv \
        --output TikTok_Acne_Summarized.csv --api-key sk-ant-...
    
    # Test
    python transcript_summary.py --input data.csv --output test.csv --api-key KEY --limit 5
"""

import csv
import json
import time
import argparse
import sys

try:
    import anthropic
except ImportError:
    print("ERROR: pip install anthropic")
    sys.exit(1)


SUMMARY_PROMPT = """Summarize this TikTok video transcript about acne/skincare in 1-2 sentences.
Focus on: what treatment or product is mentioned, what claim is made, and what outcome is described.
If the transcript is nonsensical, just music, or unrelated to skincare, respond with "Not relevant".
Respond with ONLY the summary, no labels or prefixes."""


def summarize_transcript(client, transcript):
    """Summarize a single transcript."""
    if not transcript or len(transcript.split()) < 10:
        return ""
    
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": f"{SUMMARY_PROMPT}\n\nTranscript: {transcript[:1000]}"
            }]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"  Error: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(description="Summarize video transcripts")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    client = anthropic.Anthropic(api_key=args.api_key)

    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if "transcript_summary" not in fieldnames:
        fieldnames.append("transcript_summary")

    total = min(len(rows), args.limit) if args.limit else len(rows)
    summarized = 0

    print(f"Summarizing {total} transcripts...")

    for i, row in enumerate(rows):
        if args.limit and i >= args.limit:
            break
        
        if args.resume and row.get("transcript_summary", "").strip():
            continue

        transcript = row.get("transcript", "").strip()
        if not transcript or len(transcript.split()) < 10:
            row["transcript_summary"] = ""
            continue

        print(f"  [{i+1}/{total}]...", end=" ", flush=True)
        
        summary = summarize_transcript(client, transcript)
        row["transcript_summary"] = summary
        summarized += 1
        
        print(f"OK ({len(summary)} chars)" if summary else "skipped")

        # Checkpoint
        if (i + 1) % 20 == 0:
            with open(args.output, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

        time.sleep(0.3)

    # Final save
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Summarized {summarized} transcripts → {args.output}")


if __name__ == "__main__":
    main()
