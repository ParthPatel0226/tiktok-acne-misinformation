"""
Final Derivations Cell (Cell 5 in Colab)
========================================
8 rule-based derivations computed from existing columns.
Run after all NLP classification is done.

Usage (standalone):
    python final_derivations.py --input TikTok_Acne_Classified.csv --output TikTok_Acne_Derived.csv

Usage (Colab): paste contents into a cell after NLP classification cells.
"""

import csv
import re
import argparse
import sys


# ── 1. Condition Recurrence ──────────────────────────────────────────

RECUR_YES_KW = ["keep coming back", "recurring", "chronic", "years", "always",
                "never goes away", "constant", "persistent", "lifelong", "flare"]
RECUR_NO_KW = ["first time", "new breakout", "sudden", "recently", "just started",
               "never had", "one-time"]


def derive_recurrence(row):
    text = (row.get("caption", "") + " " + row.get("transcript", "")).lower()
    dur = row.get("condition_duration", "").lower().strip()
    
    if dur == "years":
        return "yes"
    
    for kw in RECUR_YES_KW:
        if kw in text:
            return "yes"
    for kw in RECUR_NO_KW:
        if kw in text:
            return "no"
    
    return "unknown"


# ── 2. Product Brand from Product Name ───────────────────────────────

BRAND_LOOKUP = {
    "cerave": "CeraVe", "la roche-posay": "La Roche-Posay", "laroche": "La Roche-Posay",
    "neutrogena": "Neutrogena", "the ordinary": "The Ordinary", "paula's choice": "Paula's Choice",
    "differin": "Galderma", "panoxyl": "PanOxyl", "cosrx": "COSRX",
    "cetaphil": "Cetaphil", "proactiv": "Proactiv", "mario badescu": "Mario Badescu",
    "drunk elephant": "Drunk Elephant", "tatcha": "Tatcha", "glow recipe": "Glow Recipe",
    "innisfree": "Innisfree", "anua": "Anua", "purito": "PURITO",
    "some by mi": "Some By Mi", "beauty of joseon": "Beauty of Joseon",
    "skinceuticals": "SkinCeuticals", "elf": "e.l.f.", "hero cosmetics": "Hero Cosmetics",
}


def derive_brand(row):
    existing = row.get("product_brand", "").strip()
    if existing and existing.lower() not in ("", "none", "unknown", "na"):
        return existing
    
    product = row.get("product_name", "").lower().strip()
    if not product or product in ("none", "na", ""):
        return ""
    
    for key, brand in BRAND_LOOKUP.items():
        if key in product:
            return brand
    return ""


# ── 3. Active Ingredient from Product Name ───────────────────────────

PRODUCT_INGREDIENT_MAP = {
    "panoxyl": "benzoyl peroxide", "differin": "adapalene",
    "proactiv": "benzoyl peroxide", "epiduo": "adapalene + benzoyl peroxide",
    "tretinoin": "tretinoin", "retin-a": "tretinoin",
    "accutane": "isotretinoin", "isotretinoin": "isotretinoin",
    "spironolactone": "spironolactone", "clindamycin": "clindamycin",
    "salicylic": "salicylic acid", "glycolic": "glycolic acid",
    "niacinamide": "niacinamide", "azelaic": "azelaic acid",
    "benzoyl peroxide": "benzoyl peroxide", "retinol": "retinol",
    "tea tree": "tea tree oil", "turmeric": "turmeric",
    "neem": "neem", "aloe vera": "aloe vera",
}


def derive_ingredient(row):
    existing = row.get("ingredient_named", row.get("active_ingredient", "")).strip()
    if existing and existing.lower() not in ("", "none", "unknown", "na"):
        return existing
    
    text = (row.get("product_name", "") + " " + row.get("caption", "")).lower()
    
    for key, ingredient in PRODUCT_INGREDIENT_MAP.items():
        if key in text:
            return ingredient
    return ""


# ── 4. Skin Condition Type from Caption/Transcript ───────────────────

CONDITION_KEYWORDS = {
    "cystic_acne": ["cystic", "cystic acne", "deep cyst", "painful bump"],
    "hormonal_acne": ["hormonal", "hormonal acne", "period breakout", "chin acne", "jawline"],
    "acne_scars": ["acne scar", "scarring", "PIH", "hyperpigmentation", "dark spot", "ice pick"],
    "blackheads": ["blackhead", "blackheads", "sebaceous filament", "pore strip"],
    "whiteheads": ["whitehead", "whiteheads", "closed comedone"],
    "fungal_acne": ["fungal acne", "malassezia", "fungal", "pityrosporum"],
    "back_acne": ["bacne", "back acne", "body acne", "chest acne"],
    "mild_acne": ["mild acne", "few pimples", "occasional breakout"],
}


def derive_condition_type(row):
    existing = row.get("skin_condition_type", "").strip()
    if existing and existing.lower() not in ("", "none", "unknown", "na", "acne"):
        return existing
    
    text = (row.get("caption", "") + " " + row.get("transcript", "")).lower()
    
    for condition, keywords in CONDITION_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return condition
    return "acne_vulgaris"  # default


# ── 5. Country Origin from Caption Language + Keywords ───────────────

COUNTRY_KEYWORDS = {
    "India": ["india", "ayurvedic", "turmeric", "neem", "hindi", "desi", "multani mitti"],
    "South Korea": ["korea", "korean", "k-beauty", "kbeauty", "cosrx", "anua", "seoul"],
    "Japan": ["japan", "japanese", "j-beauty"],
    "China": ["chinese", "china", "tcm", "traditional chinese"],
    "Nigeria": ["nigeria", "nigerian", "african skin", "melanin"],
    "Brazil": ["brazil", "brazilian"],
    "Pakistan": ["pakistan", "pakistani"],
    "UK": ["uk", "nhs", "british", "london"],
    "Middle East": ["arabic", "arab", "middle east", "black seed", "hijab"],
}


def derive_country(row):
    existing = row.get("country_origin", "").strip()
    if existing and existing.lower() not in ("", "unknown", "na"):
        return existing
    
    text = (row.get("caption", "") + " " + row.get("transcript", "") + " " + 
            row.get("creator_bio", "")).lower()
    
    for country, keywords in COUNTRY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return country
    
    # Default to US for English content
    lang = row.get("language", row.get("audio_language", "")).lower()
    if lang in ("en", "english"):
        return "US"
    return "Unknown"


# ── Main ─────────────────────────────────────────────────────────────

def apply_all_derivations(rows):
    """Apply all 8 derivations to each row."""
    for row in rows:
        row["condition_recurrence"] = derive_recurrence(row)
        
        brand = derive_brand(row)
        if brand:
            row["product_brand"] = brand
        
        ingredient = derive_ingredient(row)
        if ingredient:
            row.setdefault("active_ingredient", "")
            if not row["active_ingredient"].strip():
                row["active_ingredient"] = ingredient
        
        row["skin_condition_type"] = derive_condition_type(row)
        row["country_origin"] = derive_country(row)
    
    return rows


def main():
    parser = argparse.ArgumentParser(description="Apply rule-based derivations to dataset")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    # Add new columns
    for col in ["condition_recurrence", "active_ingredient"]:
        if col not in fieldnames:
            fieldnames.append(col)

    rows = apply_all_derivations(rows)

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. {len(rows)} rows processed → {args.output}")


if __name__ == "__main__":
    main()
