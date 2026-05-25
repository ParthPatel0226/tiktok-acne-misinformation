# TikTok Audio Download + Transcription — Setup Guide

## Overview
This pipeline does 3 things:
1. Downloads audio from all 731 TikTok videos using yt-dlp
2. Transcribes each audio file using OpenAI Whisper (runs locally, free)
3. Saves transcripts back into your CSV for NLP classification

---

## Step 1 — Install Python dependencies

```bash
pip install yt-dlp openai-whisper
```

---

## Step 2 — Install ffmpeg

ffmpeg converts video to audio. Required by both yt-dlp and Whisper.

### Windows:
1. Go to: https://www.gyan.dev/ffmpeg/builds/
2. Download **ffmpeg-release-essentials.zip**
3. Unzip it somewhere (e.g. `C:\ffmpeg`)
4. Add `C:\ffmpeg\bin` to your system PATH:
   - Search "Environment Variables" in Start menu
   - Edit PATH → Add `C:\ffmpeg\bin`
5. Test: open PowerShell and run `ffmpeg -version`

### Mac:
```bash
brew install ffmpeg
```

### Linux:
```bash
sudo apt install ffmpeg
```

---

## Step 3 — Get TikTok cookies (important — avoids blocks)

TikTok blocks anonymous downloads. Fix this by exporting your cookies:

1. Install browser extension: **"Get cookies.txt LOCALLY"**
   - Chrome: https://chromewebstore.google.com/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc
   - Firefox: search "cookies.txt" in Firefox Add-ons
2. Go to **tiktok.com** and make sure you're logged in
3. Click the extension icon → Export cookies for tiktok.com
4. Save file as `cookies.txt` in the same folder as the script

---

## Step 4 — Run the pipeline

### Test first (5 videos):
```bash
python download_and_transcribe.py \
  --input TikTok_Acne_-_Merged_Data.csv \
  --output test_transcribed.csv \
  --cookies cookies.txt \
  --limit 5
```

### Full run (all 731 videos):
```bash
python download_and_transcribe.py \
  --input TikTok_Acne_-_Merged_Data.csv \
  --output TikTok_Acne_Transcribed.csv \
  --cookies cookies.txt \
  --model base
```

### If interrupted, resume where you left off:
```bash
python download_and_transcribe.py \
  --input TikTok_Acne_-_Merged_Data.csv \
  --output TikTok_Acne_Transcribed.csv \
  --cookies cookies.txt \
  --model base \
  --resume
```

---

## Whisper model options

| Model    | Speed (CPU) | Accuracy | RAM needed |
|----------|-------------|----------|------------|
| tiny     | ~3s/video   | Low      | ~1 GB      |
| base     | ~7s/video   | Good     | ~1 GB      | ← recommended
| small    | ~15s/video  | Better   | ~2 GB      |
| medium   | ~40s/video  | High     | ~5 GB      |
| large-v3 | ~90s/video  | Best     | ~10 GB     |

For 731 videos with `base` model: ~1.5 hours on CPU

If you have an NVIDIA GPU, it'll be 5–10x faster automatically.

---

## Estimated time and storage

- **Time**: ~1.5–2 hours for all 731 videos (base model, CPU)
- **Storage**: ~150 MB for audio files (deleted after transcription by default)
- **Cost**: Free — Whisper runs locally

---

## Common errors and fixes

| Error | Fix |
|-------|-----|
| `ffmpeg not found` | Add ffmpeg to PATH (see Step 2) |
| `HTTP Error 403` | Add cookies.txt (see Step 3) |
| `Video unavailable` | Video was deleted — script skips it automatically |
| `Rate limited` | Script waits 30s and retries automatically |
| `CUDA out of memory` | Use smaller model: `--model base` |
| Script interrupted | Run again with `--resume` flag |

---

## After transcription

Upload `TikTok_Acne_Transcribed.csv` back to Claude, then run the NLP classifier:

```bash
python classify_videos_v2.py \
  --input TikTok_Acne_Transcribed.csv \
  --output TikTok_Acne_Classified.csv \
  --api-key sk-ant-...
```

The classifier uses transcripts + captions for much better accuracy on claim detection.
