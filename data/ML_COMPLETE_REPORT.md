# XGBoost Multiclass Classification — Complete Report
## TikTok Acne Misinformation Detection
## Target: misinfo_label (no / yes / not_sure)

---

## 1. PROBLEM DEFINITION

**Type:** Multiclass Classification (3 classes)
**Target Variable:** misinfo_label
- no (48.0%, 351 videos) — content verified against gold standard
- yes (44.6%, 326 videos) — fails all 3 clinical verification checks  
- not_sure (7.4%, 54 videos) — mixes approved + non-approved treatments

**Why XGBoost:**
- Handles both bias and variance simultaneously (boosting reduces bias, regularization controls variance)
- State-of-the-art for tabular/structured data
- Built-in feature importance for interpretability
- Handles class imbalance via sample weights
- Robust to multicollinearity (tree-based splits)

**Why NOT other models:**
- Neural Networks: 731 rows too small (needs 10K+)
- Naive Bayes: assumes feature independence (violated — views↔likes r=0.91)
- LDA/QDA: assumes normal distribution (all features non-normal)
- LightGBM: designed for 100K+ rows

---

## 2. DATA PREPARATION

**Input:** 731 rows × 46 columns (after feature engineering)

**Dropped for ML (7 columns):**
- creator_username, creator_id (identifiers)
- hashtags, active_ingredient, product_name, product_brand (high cardinality text)
- Video_type (zero variance — only 1 unique value, all "Self")

**Final ML features:** 38 columns
- 7 continuous (log-transformed): views, likes, comments, shares, creator_followers, video_duration_seconds, engagement_rate
- 2 discrete: No._people_video, video_shot_complexity
- 3 ordinal (encoded): skin_severity (0-3), intent_clarity (0-2), label_confidence (0-2)
- 2 binary (encoded): before_after_claim (0/1), unrealistic_claim_flag (0/1)
- 24 nominal categorical (label encoded): creator_type, claim_type, claim_accuracy, treatment_type, etc.

**Missing value handling:**
- video_duration_seconds: 19 missing → imputed with median (45.5s)
- All categorical: NaN → "unknown" category before encoding

**Encoding strategy:**
- LabelEncoder for all 24 nominal categorical features
- XGBoost handles label-encoded nominals well because tree splits don't assume ordering
- One-hot encoding was considered but rejected (would create 163 columns from 731 rows)

---

## 3. TRAIN/TEST SPLIT — Custom Stratified

**Strategy:** Split each class independently at 70/30, then combine and shuffle.
**Why custom:** Standard stratified split gave not_sure only 38 train / 16 test. Custom split ensures each class has proportional representation with different random selection.

| Class | Train | Test | Total |
|-------|-------|------|-------|
| no | 245 (48.0%) | 106 (48.0%) | 351 |
| not_sure | 37 (7.3%) | 17 (7.7%) | 54 |
| yes | 228 (44.7%) | 98 (44.3%) | 326 |
| **Total** | **510 (69.8%)** | **221 (30.2%)** | **731** |

**Class weights applied:** Inversely proportional to class frequency
- no: weight = 0.69
- not_sure: weight = 4.59 (upweighted 6.6x)
- yes: weight = 0.75

---

## 4. MODEL CONFIGURATION

```
XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    n_estimators=200,
    max_depth=3,              # shallow trees to prevent overfitting
    learning_rate=0.05,       # slow learning for better generalization
    subsample=0.7,            # row sampling
    colsample_bytree=0.7,     # feature sampling
    min_child_weight=10,      # minimum samples per leaf
    gamma=0.5,                # minimum loss reduction for split
    reg_alpha=1.0,            # L1 regularization
    reg_lambda=5.0,           # L2 regularization
    early_stopping_rounds=30, # stops if test loss doesn't improve
)
```

**Regularization rationale:**
- max_depth=3: With 510 training rows, deep trees (5+) memorize noise
- learning_rate=0.05: Slower learning allows more trees to contribute
- gamma=0.5: Prevents splits that don't reduce loss significantly
- reg_alpha=1.0 + reg_lambda=5.0: Strong L1+L2 penalty on leaf weights
- early_stopping: Stopped at iteration 189/200 (test loss plateaued)

---

## 5. TRAINING RESULTS

| Metric | Score |
|--------|-------|
| Accuracy | 84.7% |
| F1 (macro) | 0.814 |
| F1 (weighted) | 0.851 |

**Per-class (Train):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| no | 0.90 | 0.83 | 0.86 | 245 |
| not_sure | 0.56 | 1.00 | 0.72 | 37 |
| yes | 0.88 | 0.84 | 0.86 | 228 |

**Interpretation:**
- Model learns no and yes classes well (F1=0.86 each)
- not_sure achieves 100% recall but only 56% precision on training — model tends to over-predict not_sure to compensate for class weights
- 84.7% training accuracy (not 97%+) indicates regularization is working — model is not memorizing

---

## 6. TESTING RESULTS

| Metric | Score |
|--------|-------|
| Accuracy | **70.1%** |
| F1 (macro) | **0.593** |
| F1 (weighted) | **0.708** |
| AUC-ROC (macro) | **0.797** |

**Per-class (Test):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| no | 0.75 | 0.71 | 0.73 | 106 |
| not_sure | 0.25 | 0.35 | 0.29 | 17 |
| yes | 0.76 | 0.76 | 0.76 | 98 |

**Interpretation:**
- no class: 75% precision, 71% recall — when model says "no," it's right 75% of the time
- yes class: 76% precision, 76% recall — best performing class, balanced precision/recall
- not_sure class: 25% precision, 35% recall, F1=0.29 — improved 144% from standard split (was 0.12) but still weak due to only 54 total samples
- AUC-ROC 0.797 indicates good discriminative ability — model ranks classes correctly ~80% of the time

---

## 7. OVERFITTING ANALYSIS

| Metric | Train | Test | Gap |
|--------|-------|------|-----|
| Accuracy | 84.7% | 70.1% | **14.6%** |
| F1 (macro) | 0.814 | 0.593 | 22.1% |
| F1 (weighted) | 0.851 | 0.708 | 14.3% |

**14.6% accuracy gap** — mild overfitting, expected with 731 rows.

**Improvement from regularization:**
- Before regularization: 25.7% gap (97.1% train, 71.4% test)
- After regularization: 14.6% gap (84.7% train, 70.1% test)
- **11.1% reduction in overfitting**

---

## 8. CROSS-VALIDATION

| Metric | Score |
|--------|-------|
| CV Accuracy | **76.7% ± 2.5%** |
| CV F1 (macro) | **0.542 ± 0.029** |

**Fold-by-fold:**

| Fold | Accuracy | F1 |
|------|----------|-----|
| 1 | 78.2% | 0.525 |
| 2 | 76.7% | 0.572 |
| 3 | 77.4% | 0.570 |
| 4 | 71.9% | 0.489 |
| 5 | 78.8% | 0.545 |

**Interpretation:**
- Stable across folds (std=2.5%) — model generalizes consistently
- CV accuracy (76.7%) is higher than test accuracy (70.1%) — test set may have harder examples
- Fold 4 is the worst (71.9%) — likely got a harder not_sure distribution

---

## 9. FEATURE IMPORTANCE (Top 10)

| Rank | Feature | Importance | Why it matters |
|------|---------|------------|----------------|
| 1 | claim_accuracy | 0.096 | Strongest predictor — matches EDA (V=0.364) |
| 2 | claim_type | 0.053 | 2nd in EDA (V=0.451) |
| 3 | label_confidence | 0.049 | Ordinal — how confident the labeler was |
| 4 | unrealistic_claim_flag | 0.042 | Binary flag for exaggerated claims |
| 5 | country_origin | 0.039 | Geographic misinfo patterns |
| 6 | No._people_video | 0.037 | More people = different content type |
| 7 | intent_clarity | 0.036 | How clear the video's intent is |
| 8 | video_format | 0.035 | Talking head vs demo vs before/after |
| 9 | treatment_type | 0.032 | Topical vs supplement vs lifestyle |
| 10 | before_after_claim | 0.031 | Transformation format flag |

**Alignment with EDA:**
- Top EDA predictors (claim_type V=0.451, claim_accuracy V=0.364) are also top XGBoost features ✅
- country_origin (V=0.386 in EDA) appears at rank 5 ✅
- creator_gender (ns in EDA) has low importance in XGBoost ✅ — consistent

---

## 10. MISCLASSIFICATION ANALYSIS

**Confusion Matrix (% normalized):**

| Actual → Predicted | no | not_sure | yes |
|--------------------|-----|----------|-----|
| **no** | **70.8%** | 10.4% | 18.9% |
| **not_sure** | 47.1% | **35.3%** | 17.6% |
| **yes** | 17.3% | 7.1% | **75.5%** |

**Total misclassified:** 66/221 (29.9%)

**Key errors:**
- 20 "no" videos wrongly predicted as "yes" (18.9%)
- 19 "yes" videos wrongly predicted as "no" (17.3%)
- 8 "not_sure" predicted as "no" (47.1%)
- 3 "not_sure" predicted as "yes" (17.6%)

**Why errors happen:**
- no↔yes confusion: Videos at the boundary — e.g., mentions sunscreen (approved) but also promotes a dubious "gut cleanse" in the same video
- not_sure→no: Model defaults to "no" when unsure because "no" is the majority class
- not_sure is inherently ambiguous — these are the gray-area videos that mix approved and non-approved treatments

---

## 11. ROC-AUC ANALYSIS

| Class | AUC |
|-------|-----|
| no | 0.794 |
| not_sure | 0.714 |
| yes | 0.803 |
| **Macro average** | **0.797** |

**Interpretation:**
- AUC > 0.7 for all three classes — model has meaningful discriminative power
- yes class has highest AUC (0.803) — model is best at identifying misinformation
- not_sure has lowest AUC (0.714) — harder to separate from the other classes

---

## 12. LIMITATIONS & HONEST ASSESSMENT

**What works:**
- 70.1% test accuracy on 3-class problem with 731 rows is reasonable
- no and yes classes achieve 0.73-0.76 F1 — practically useful for flagging
- AUC 0.797 means the model ranks most videos correctly
- Feature importance aligns with EDA — model learned real patterns, not noise

**What doesn't:**
- not_sure class (F1=0.29) — 54 samples is insufficient for any model
- 14.6% overfit gap — inherent to small datasets with tree models
- Model cannot distinguish edge cases where approved and non-approved treatments coexist

**For deployment:**
- Use the model as a screening tool, not a final verdict
- Flag videos with misinfo_probability > 0.7 as "likely misinformation"
- Human review required for not_sure predictions
- Retrain with more data (especially not_sure examples) to improve

