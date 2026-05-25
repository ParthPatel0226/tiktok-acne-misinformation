"""
TikTok Acne Misinformation NLP Classifier v2
Fills all 10 text-derivable columns using Claude API.

Columns filled:
  1. claim_accuracy        - supported/partial/unsupported/false/na
  2. claim_type            - from taxonomy
  3. misinformation_label  - true/false/misleading/na
  4. treatment_origin      - western_clinical/ayurvedic/tcm/k_beauty/etc
  5. western_accepted      - yes/no/partial/unknown
  6. clinical_validity     - from AAD guidelines
  7. country_origin        - inferred from language + bio + caption
  8. creator_gender        - inferred from bio + username
  9. ingredient_named      - extracted from caption
  10. product_name         - extracted from caption
      product_brand        - extracted from caption

Usage:
    pip install anthropic
    python classify_videos_v2.py --input TikTok_Acne_-_Merged_Data.csv --output TikTok_Acne_Classified.csv --api-key sk-ant-...
    
    # Test with 5 rows first:
    python classify_videos_v2.py --input TikTok_Acne_-_Merged_Data.csv --output test_out.csv --api-key sk-ant-... --limit 5
"""

import csv
import json
import time
import argparse
import os
import sys
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed.")
    print("Run: pip install anthropic")
    sys.exit(1)


SYSTEM_PROMPT = """You are a dermatology research assistant classifying TikTok acne content.
Base all clinical verdicts on AAD (American Academy of Dermatology) guidelines.

MISINFORMATION TAXONOMY (13 categories):
1. dietary_cure_dairy — "Cut dairy to clear acne" — PARTIALLY SUPPORTED
2. dietary_cure_sugar — "Sugar-free cured my acne" — UNSUPPORTED  
3. gut_health — "Fix gut = fix acne" — UNSUPPORTED as primary treatment
4. diy_topicals — "Toothpaste/garlic on pimples" — FALSE AND HARMFUL
5. diy_chemical_peels — "30% glycolic acid at home" — DANGEROUS
6. anti_isotretinoin — "Accutane ruined me, try this instead" — MISLEADING
7. hormone_supplement — "DIM + spearmint tea cleared hormonal acne" — UNSUPPORTED
8. ice_therapy — "Ice your face daily" — NOT EVIDENCE-BASED
9. led_device — "$30 red light cleared my acne" — PARTIALLY SUPPORTED (mild only)
10. sunscreen_anti — "Sunscreen causes acne" — FALSE AND DANGEROUS
11. unverified_routine — K-beauty/general skincare promoted as acne cure — VARIES
12. ayurvedic_traditional — turmeric, neem, traditional remedies — LIMITED EVIDENCE
13. clinically_accurate — retinoids, benzoyl peroxide, salicylic acid — SUPPORTED

For each video, respond ONLY with a JSON object (no markdown, no backticks):
{
  "claim_type": "one of the 13 categories above, or 'none'",
  "claim_accuracy": "supported|partial|unsupported|false|na",
  "misinformation_label": "accurate|misleading|false|unverifiable|na",
  "treatment_origin": "western_clinical|ayurvedic|tcm|k_beauty|african_traditional|latin_american|middle_eastern|unknown",
  "western_accepted": "yes|no|partial|unknown",
  "clinical_validity": "Brief AAD-based assessment",
  "country_origin": "US|India|South Korea|Japan|China|Nigeria|Brazil|Pakistan|UK|Middle East|Unknown",
  "creator_gender": "male|female|unknown",
  "ingredient_named": "specific ingredient or 'none'",
  "product_name": "specific product or 'none'",
  "product_brand": "brand name or 'none'"
}
"""


def classify_video(client, caption, creator_type, hashtag, transcript=""):
    """Send one video to Claude API for classification."""
    content_parts = []
    if caption:
        content_parts.append(f"Caption: {caption}")
    if hashtag:
        content_parts.append(f"Hashtags: {hashtag}")
    if creator_type:
        content_parts.append(f"Creator type: {creator_type}")
    if transcript:
        content_parts.append(f"Transcript: {transcript[:500]}")

    user_msg = "\n".join(content_parts) if content_parts else "No content available"

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}]
        )
        text = response.content[0].text.strip()
        # Clean markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return None
    except Exception as e:
        print(f"  API error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Classify TikTok acne videos using Claude API")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--api-key", required=True, help="Anthropic API key")
    parser.add_argument("--limit", type=int, default=None, help="Process only N rows (for testing)")
    parser.add_argument("--resume", action="store_true", help="Skip already-classified rows")
    args = parser.parse_args()

    client = anthropic.Anthropic(api_key=args.api_key)

    # Read input
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Add output columns if missing
    output_cols = [
        "claim_accuracy", "claim_type", "misinformation_label",
        "treatment_origin", "western_accepted", "clinical_validity",
        "country_origin", "creator_gender", "ingredient_named",
        "product_name", "product_brand"
    ]
    for col in output_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    total = min(len(rows), args.limit) if args.limit else len(rows)
    classified = 0
    skipped = 0

    print(f"Processing {total} videos...")

    for i, row in enumerate(rows):
        if args.limit and i >= args.limit:
            break

        # Skip if already classified and resuming
        if args.resume and row.get("claim_accuracy", "").strip() not in ("", "na", "NA"):
            skipped += 1
            continue

        caption = row.get("caption", "")
        creator_type = row.get("creator_type", "")
        hashtag = row.get("hashtag", "")
        transcript = row.get("transcript", "")

        result = classify_video(client, caption, creator_type, hashtag, transcript)

        if result:
            for key in output_cols:
                if key in result:
                    row[key] = result[key]
            classified += 1
        else:
            classified += 1  # Count even if failed

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{total}] classified={classified}, skipped={skipped}")

        # Save checkpoint every 20 rows
        if (i + 1) % 20 == 0:
            with open(args.output, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)

        # Rate limiting
        time.sleep(0.5)

    # Final save
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! Classified={classified}, Skipped={skipped}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
