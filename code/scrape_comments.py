"""
TikTok Comments Scraper via Apify
Scrapes top comments per video using Apify Comments Scraper actor.

Usage:
    python scrape_comments.py --input TikTok_Acne_Transcribed.csv \
        --api-key YOUR_APIFY_TOKEN \
        --output-comments tiktok_comments.csv \
        --output-main TikTok_Acne_With_Comments.csv

    # Test on 5 videos
    python scrape_comments.py --input data.csv --api-key TOKEN --limit 5

    # Limit comments per video
    python scrape_comments.py --input data.csv --api-key TOKEN --comments-per-video 20
"""

import csv
import json
import time
import argparse
import sys
import os

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)


APIFY_ACTOR = "clockworks/tiktok-comments-scraper"
APIFY_BASE = "https://api.apify.com/v2"


def start_apify_run(video_urls, api_key, comments_per_video=50):
    """Start an Apify comments scraper run."""
    url = f"{APIFY_BASE}/acts/{APIFY_ACTOR}/runs?token={api_key}&memory=1024&timeout=300"
    body = {
        "postURLs": video_urls,
        "commentsPerPost": comments_per_video,
        "shouldDownloadSubtitles": False,
    }
    resp = requests.post(url, json=body)
    resp.raise_for_status()
    return resp.json()["data"]


def poll_run(run_id, api_key, poll_interval=30, max_polls=40):
    """Poll Apify run until completion."""
    url = f"{APIFY_BASE}/actor-runs/{run_id}?token={api_key}"
    for attempt in range(max_polls):
        resp = requests.get(url)
        resp.raise_for_status()
        status = resp.json()["data"]["status"]
        if status == "SUCCEEDED":
            return resp.json()["data"]
        elif status in ("FAILED", "TIMED-OUT", "ABORTED"):
            raise RuntimeError(f"Apify run {status}")
        time.sleep(poll_interval)
    raise TimeoutError(f"Run did not complete after {max_polls * poll_interval}s")


def fetch_results(dataset_id, api_key):
    """Fetch results from Apify dataset."""
    url = f"{APIFY_BASE}/datasets/{dataset_id}/items?format=json&token={api_key}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Scrape TikTok comments via Apify")
    parser.add_argument("--input", required=True, help="Input CSV with video_url column")
    parser.add_argument("--api-key", required=True, help="Apify API token")
    parser.add_argument("--comments-per-video", type=int, default=50, help="Max comments per video")
    parser.add_argument("--batch-size", type=int, default=20, help="Videos per Apify run")
    parser.add_argument("--limit", type=int, default=None, help="Process only N videos")
    parser.add_argument("--resume", action="store_true", help="Skip videos with existing comments")
    parser.add_argument("--acne-only", action="store_true", help="Only scrape acne-relevant videos")
    parser.add_argument("--output-comments", default="tiktok_comments.csv", help="Comments output file")
    parser.add_argument("--output-main", default=None, help="Updated main CSV with top_comments column")
    args = parser.parse_args()

    if not args.output_main:
        base = os.path.splitext(args.input)[0]
        args.output_main = f"{base}_with_comments.csv"

    # Read input
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    # Add top_comments column
    if "top_comments" not in fieldnames:
        fieldnames.append("top_comments")

    # Filter rows
    urls_to_scrape = []
    for row in rows:
        if args.acne_only and row.get("is_acne_relevant", "").lower() != "yes":
            continue
        if args.resume and row.get("top_comments", "").strip():
            continue
        url = row.get("video_url", "").strip()
        if url:
            urls_to_scrape.append(url)

    if args.limit:
        urls_to_scrape = urls_to_scrape[:args.limit]

    print(f"Videos to scrape: {len(urls_to_scrape)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Comments per video: {args.comments_per_video}")

    # Process in batches
    all_comments = []
    comment_fieldnames = ["video_url", "video_id", "comment_text", "comment_likes",
                          "comment_username", "comment_date"]

    for batch_start in range(0, len(urls_to_scrape), args.batch_size):
        batch = urls_to_scrape[batch_start:batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        total_batches = (len(urls_to_scrape) + args.batch_size - 1) // args.batch_size

        print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} videos)...")

        try:
            run_data = start_apify_run(batch, args.api_key, args.comments_per_video)
            run_id = run_data["id"]
            dataset_id = run_data["defaultDatasetId"]
            print(f"  Run started: {run_id}")

            completed = poll_run(run_id, args.api_key)
            print(f"  Run completed. Fetching results...")

            results = fetch_results(dataset_id, args.api_key)
            
            # Process comments
            batch_comments = 0
            for item in results:
                video_url = item.get("videoUrl", item.get("url", ""))
                video_id = item.get("videoId", "")
                comments = item.get("comments", [])
                
                # Map back to main dataset
                for row in rows:
                    if row.get("video_url", "") == video_url or row.get("video_id", "") == str(video_id):
                        top_5 = [c.get("text", "") for c in comments[:5] if c.get("text")]
                        row["top_comments"] = " | ".join(top_5)
                        break

                for comment in comments:
                    all_comments.append({
                        "video_url": video_url,
                        "video_id": video_id,
                        "comment_text": comment.get("text", ""),
                        "comment_likes": comment.get("likes", 0),
                        "comment_username": comment.get("username", ""),
                        "comment_date": comment.get("createTime", ""),
                    })
                    batch_comments += 1

            print(f"  Collected {batch_comments} comments")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # Save progress after each batch
        with open(args.output_comments, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=comment_fieldnames)
            writer.writeheader()
            writer.writerows(all_comments)
        
        with open(args.output_main, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"  [Saved: {len(all_comments)} total comments]\n")
        time.sleep(5)  # Pause between batches

    print(f"\n{'='*55}")
    print(f"COMPLETE")
    print(f"  Total comments collected: {len(all_comments)}")
    print(f"  Comments file: {args.output_comments}")
    print(f"  Updated main dataset: {args.output_main}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
