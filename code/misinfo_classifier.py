"""
Misinformation Classifier — AAD Ground Truth Comparison
========================================================
Classifies each video's treatment claim against the AAD/FDA approved treatment list.
Three verification checks determine final misinformation label.

Usage:
    python misinfo_classifier.py --input TikTok_Acne_Derived.csv --output TikTok_Acne_Final.csv
"""

import csv
import re
import argparse
import sys


# ── AAD + FDA Approved Acne Treatments ───────────────────────────────

APPROVED_TREATMENTS = {
    # Topical treatments
    "benzoyl peroxide", "salicylic acid", "adapalene", "tretinoin",
    "tazarotene", "clindamycin", "erythromycin", "dapsone",
    "azelaic acid", "sulfacetamide", "trifarotene",
    
    # Oral treatments
    "isotretinoin", "doxycycline", "minocycline", "tetracycline",
    "sarecycline", "spironolactone", "oral contraceptives",
    
    # Procedures
    "chemical peel", "light therapy", "photodynamic therapy",
    "cortisone injection", "extraction",
    
    # OTC approved
    "retinol", "niacinamide", "sulfur", "glycolic acid",
    "lactic acid", "hydrocolloid patch",
}

APPROVED_INGREDIENTS = {
    "benzoyl peroxide", "salicylic acid", "adapalene", "tretinoin",
    "retinol", "niacinamide", "azelaic acid", "glycolic acid",
    "sulfur", "clindamycin", "erythromycin", "dapsone",
    "isotretinoin", "spironolactone", "doxycycline",
    "lactic acid", "tea tree oil",  # limited evidence but commonly discussed
}

# Treatments that are NOT approved / evidence-based
NON_APPROVED = {
    "turmeric", "neem", "rice water", "black seed oil", "coconut oil",
    "yogurt", "lemon juice", "baking soda", "apple cider vinegar",
    "toothpaste", "garlic", "honey", "aloe vera", "tea tree",
    "multani mitti", "leech", "snail mucin", "bee venom",
    "gut health", "probiotics", "seed oils", "DIM",
    "spearmint tea", "collagen", "bone broth",
}

# Debunking keywords — if present, the video is educational, not misinformation
DEBUNK_KEYWORDS = [
    "don't do this", "myth", "debunk", "actually doesn't work",
    "dermatologist reacts", "stop doing this", "this is wrong",
    "not recommended", "harmful", "dangerous", "please don't",
    "no evidence", "not proven", "misinformation",
]


def check_treatment_approved(treatment_type, active_ingredient):
    """Check 1: Is the treatment on the AAD/FDA approved list?"""
    treatment = (treatment_type or "").lower().strip()
    ingredient = (active_ingredient or "").lower().strip()
    
    if not treatment and not ingredient:
        return "unknown"
    
    # Check against approved lists
    for approved in APPROVED_TREATMENTS:
        if approved in treatment or approved in ingredient:
            return "approved"
    
    for approved in APPROVED_INGREDIENTS:
        if approved in ingredient:
            return "approved"
    
    # Check against non-approved
    for non in NON_APPROVED:
        if non in treatment or non in ingredient:
            return "not_approved"
    
    return "unknown"


def check_is_debunking(caption, transcript):
    """Check if the video is debunking/educating against the treatment."""
    text = ((caption or "") + " " + (transcript or "")).lower()
    
    for kw in DEBUNK_KEYWORDS:
        if kw in text:
            return True
    return False


def classify_misinfo(row):
    """Apply three verification checks to determine misinformation label."""
    treatment_type = row.get("treatment_type", "")
    active_ingredient = row.get("active_ingredient", row.get("ingredient_named", ""))
    claim_accuracy = row.get("claim_accuracy", "")
    caption = row.get("caption", "")
    transcript = row.get("transcript", "")
    creator_type = row.get("creator_type", "")
    
    # Check 1: Treatment on approved list?
    approval_status = check_treatment_approved(treatment_type, active_ingredient)
    
    # Check 2: Is this debunking content?
    is_debunking = check_is_debunking(caption, transcript)
    
    # Check 3: Does claim_accuracy align?
    accuracy = (claim_accuracy or "").lower().strip()
    
    # Decision logic
    if is_debunking:
        return "accurate", "Debunking/educational content about non-approved treatment", ""
    
    if approval_status == "approved":
        return "accurate", "Promotes AAD/FDA approved treatment", ""
    
    if approval_status == "not_approved":
        # Find which treatment was flagged
        flagged = ""
        text = (treatment_type + " " + (active_ingredient or "")).lower()
        for non in NON_APPROVED:
            if non in text:
                flagged = non
                break
        
        if accuracy in ("false", "unsupported"):
            return "misinformation", f"Promotes non-approved treatment: {flagged}", flagged
        elif accuracy == "partial":
            return "partial", f"Limited evidence treatment: {flagged}", flagged
        else:
            return "misinformation", f"Promotes non-approved treatment: {flagged}", flagged
    
    # Unknown treatment — check claim accuracy
    if accuracy in ("supported",):
        return "accurate", "Claim accuracy rated as supported", ""
    elif accuracy in ("false",):
        return "misinformation", "Claim accuracy rated as false", ""
    elif accuracy in ("unsupported",):
        return "partial", "Claim accuracy rated as unsupported", ""
    elif accuracy in ("partial",):
        return "partial", "Claim accuracy rated as partial", ""
    
    return "unverifiable", "Unable to determine from available data", ""


def main():
    parser = argparse.ArgumentParser(description="Classify misinformation against AAD guidelines")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    # Add output columns
    for col in ["misinfo_label", "misinfo_reason", "treatment_flagged"]:
        if col not in fieldnames:
            fieldnames.append(col)

    # Classify each row
    stats = {"accurate": 0, "misinformation": 0, "partial": 0, "unverifiable": 0}
    
    for row in rows:
        label, reason, flagged = classify_misinfo(row)
        row["misinfo_label"] = label
        row["misinfo_reason"] = reason
        row["treatment_flagged"] = flagged
        stats[label] = stats.get(label, 0) + 1

    # Save
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    print(f"Misinformation Classification Complete")
    print(f"  Total:          {total}")
    print(f"  Accurate:       {stats['accurate']} ({stats['accurate']/total*100:.1f}%)")
    print(f"  Misinformation: {stats['misinformation']} ({stats['misinformation']/total*100:.1f}%)")
    print(f"  Partial:        {stats['partial']} ({stats['partial']/total*100:.1f}%)")
    print(f"  Unverifiable:   {stats['unverifiable']} ({stats['unverifiable']/total*100:.1f}%)")
    print(f"  Output:         {args.output}")


if __name__ == "__main__":
    main()
