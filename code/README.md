# TikTok Acne Misinformation Research — Code Package

All Python scripts and workflow files used in this project.

## Files

| File | Phase | Description |
|------|-------|-------------|
| `tiktok_apify_workflow_v8.json` | 1 - Scraping | n8n workflow JSON (9 nodes) — Apify → Transform → Google Sheets |
| `classify_videos_v2.py` | 2 - NLP | Claude API classifier — fills 11 labeling columns from captions |
| `download_and_transcribe.py` | 3 - Transcription | yt-dlp download + Whisper audio-to-text |
| `transcript_summary.py` | 3 - Transcription | Summarize transcripts to 1-2 sentences via Claude API |
| `scrape_comments.py` | 5 - Comments | Apify TikTok comments scraper |
| `video_analytics.py` | 4 - Vision | Gemini 2.5 Flash on video frames for 12 visual features |
| `final_derivations.py` | 5 - Enrichment | 8 rule-based derivations (recurrence, brand, ingredient, etc.) |
| `misinfo_classifier.py` | 5 - Classification | AAD/FDA approved treatment list comparison |
| `acne_scraper_v2.py` | Reference | Scrapes AAD + NLM clinical guidelines for ground truth |
| `run_pipeline.py` | Orchestration | Master runner — chains transcribe → comments → classify |

## Pipeline Order

```
Phase 1: n8n workflow (tiktok_apify_workflow_v8.json)
    ↓ 731 videos scraped to Google Sheets
Phase 2: classify_videos_v2.py
    ↓ 11 NLP columns filled
Phase 3: download_and_transcribe.py → transcript_summary.py
    ↓ 676/731 transcripts + summaries
Phase 4: video_analytics.py
    ↓ 12 visual features from video frames
Phase 5: scrape_comments.py → final_derivations.py → misinfo_classifier.py
    ↓ Comments, rule-based derivations, AAD comparison
```

## Requirements

```bash
pip install anthropic yt-dlp openai-whisper google-generativeai Pillow requests beautifulsoup4 python-docx
```

Also needed: ffmpeg (for video/audio processing)

## API Keys Needed

- **Anthropic** (Claude API): console.anthropic.com — ~$0.10-$4 total
- **Apify** (TikTok scraping): console.apify.com — ~$5-10 total
- **Google Gemini** (Vision): aistudio.google.com — FREE tier
