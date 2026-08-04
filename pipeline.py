#!/usr/bin/env python3
"""
Faceless YouTube Shorts factory: generates a facts/trivia script with Claude,
narrates it with macOS text-to-speech, pulls matching stock footage from
Pexels, burns in captions, and renders a finished vertical .mp4 ready to
upload by hand.

Usage:
    python3 pipeline.py
    python3 pipeline.py --topic "deep sea creatures"
"""
import argparse
import json
import os
import random
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY")
AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION")
AZURE_VOICE_NAME = os.environ.get("AZURE_VOICE_NAME", "en-US-OnyxTurboMultilingualNeural")
# Speaking rate relative to the voice's default. "+0%" = normal, "+15%" = 15% faster.
SPEECH_RATE = os.environ.get("SPEECH_RATE", "+15%")
MODEL = "claude-sonnet-5"

USED_TOPICS_PATH = ROOT / "state" / "used_topics.json"
OUTPUT_DIR = ROOT / "output"
MUSIC_DIR = ROOT / "music"
MUSIC_VOLUME = float(os.environ.get("MUSIC_VOLUME", "0.15"))


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# Branding
BRAND_HANDLE = os.environ.get("BRAND_HANDLE", "@TheFoundationTheory")
BRAND_SPOKEN = os.environ.get("BRAND_SPOKEN", "The Foundation Theory")
BRAND_TAGLINE = os.environ.get("BRAND_TAGLINE", "Your daily dose of weird things you didn't know")
ACCENT_COLOR = os.environ.get("ACCENT_COLOR", "#3BA7FF")
ACCENT_RGB = _hex_to_rgb(ACCENT_COLOR)
CAPTION_COLOR = os.environ.get("CAPTION_COLOR", "#FFDD00")  # caption text fill (yellow)
CAPTION_RGB = _hex_to_rgb(CAPTION_COLOR)
# Caption pacing: how many words show on screen at once (smaller = faster-moving).
CAPTION_WORDS_PER_CHUNK = int(os.environ.get("CAPTION_WORDS_PER_CHUNK", "4"))
def _resolve_font():
    # Works on both this Mac and the Linux CI runner (where the workflow
    # installs DejaVu). First existing path wins; FONT_PATH env var overrides.
    for candidate in (
        os.environ.get("FONT_PATH"),
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",           # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",        # Debian/Ubuntu
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if candidate and os.path.exists(candidate):
            return candidate
    return "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


FONT_PATH = _resolve_font()
TOKEN_PATH = ROOT / "token.json"
YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
YOUTUBE_CATEGORY_ID = os.environ.get("YOUTUBE_CATEGORY_ID", "27")  # Education

W, H = 1080, 1920


def die(msg):
    print(f"\n[ERROR] {msg}\n", file=sys.stderr)
    sys.exit(1)


def check_deps():
    for tool in ("ffmpeg", "ffprobe"):
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            die(f"'{tool}' not found on PATH. Install it before running the pipeline.")
    if not ANTHROPIC_API_KEY:
        die("ANTHROPIC_API_KEY missing. Copy .env.example to .env and fill it in.")
    if not PEXELS_API_KEY:
        die("PEXELS_API_KEY missing. Copy .env.example to .env and fill it in.")
    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        die("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION missing. Copy .env.example to .env and fill it in.")


def load_used_topics():
    if USED_TOPICS_PATH.exists():
        return json.loads(USED_TOPICS_PATH.read_text())
    return []


def save_used_topic(topic):
    used = load_used_topics()
    used.append(topic)
    USED_TOPICS_PATH.write_text(json.dumps(used[-200:], indent=2))


def generate_script(topic_override=None):
    from anthropic import Anthropic

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    used_topics = load_used_topics()
    avoid_note = ""
    if used_topics:
        avoid_note = "Avoid repeating these already-used topics: " + ", ".join(used_topics[-30:])

    topic_instruction = (
        f'The topic must be: "{topic_override}".'
        if topic_override
        else "Pick a single surprising, specific fact-based topic (science, space, history, "
        "ocean, psychology, or animals) that a general audience would find genuinely surprising. "
        + avoid_note
    )

    prompt = f"""You are writing a script for a faceless YouTube Shorts video (facts/trivia niche).

{topic_instruction}

Requirements:
- Narration: 120-150 words total, punchy short sentences, written to be read aloud by
  text-to-speech (no emoji, no markdown).
- The FIRST segment's text MUST open with a "Did you know" hook — e.g. "Did you know that..."
  or "Did you know..." — phrased as the surprising fact of this video. Keep it natural and
  varied in wording after those opening words; do not use the exact same sentence every time.
- The LAST segment MUST end with a short, punchy engagement question that invites viewers to
  reply in the comments (e.g. "Would you have risked it? Tell me below."). Do NOT include any
  "follow", "subscribe", "like", or channel call-to-action anywhere — that is added separately.
- Break the narration into 5-8 segments. Each segment is 1-2 sentences that would take
  roughly 4-8 seconds to say aloud.
- For each segment, give a short (2-4 word) visual search keyword describing stock footage
  that would visually match that segment (concrete, filmable things - e.g. "ocean waves",
  "starry night sky", "brain scan animation" - not abstract concepts).
- Write a punchy YouTube title (under 60 characters, no clickbait lies).
- Write a short YouTube description (2-3 sentences) ending with 4-6 relevant hashtags.
- Provide 8-12 relevant YouTube tags as an array of short keyword strings.
- Also return the topic you chose as "topic" (a short phrase, for tracking repeats).

Respond with ONLY valid JSON, no markdown fences, in exactly this shape:
{{
  "topic": "...",
  "title": "...",
  "description": "...",
  "tags": ["...", "..."],
  "segments": [{{"text": "...", "keyword": "..."}}]
}}"""

    resp = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    text_blocks = [block.text for block in resp.content if block.type == "text"]
    if not text_blocks:
        die(f"Claude returned no text (stop_reason={resp.stop_reason}). Try again.")
    raw = text_blocks[0].strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def synthesize_segment(text, idx, workdir):
    mp3_path = workdir / f"seg_{idx:02d}.mp3"
    wav_path = workdir / f"seg_{idx:02d}.wav"

    ssml = (
        "<speak version='1.0' xml:lang='en-US'>"
        f"<voice xml:lang='en-US' name='{AZURE_VOICE_NAME}'>"
        f"<prosody rate='{SPEECH_RATE}'>{xml_escape(text)}</prosody>"
        "</voice></speak>"
    )
    url = f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-24khz-96kbitrate-mono-mp3",
        "User-Agent": "shorts-factory",
    }
    # Azure's free (F0) tier has a low burst rate limit; firing all segments
    # back-to-back can trip a 429. Retry with backoff (honoring Retry-After)
    # so a transient rate-limit or 5xx doesn't kill the whole run.
    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        resp = requests.post(url, headers=headers, data=ssml.encode("utf-8"), timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == max_attempts:
                resp.raise_for_status()
            wait = float(resp.headers.get("Retry-After", 0)) or min(2 ** attempt, 30)
            print(f"   Azure {resp.status_code}, retrying in {wait:.0f}s "
                  f"(attempt {attempt}/{max_attempts})...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    mp3_path.write_bytes(resp.content)

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3_path),
         "-ar", "44100", "-ac", "2", str(wav_path)],
        check=True,
    )
    duration = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(wav_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip())
    return wav_path, duration


def fetch_pexels_clip(keyword, idx, workdir):
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": keyword, "orientation": "portrait", "per_page": 5, "size": "medium"}
    resp = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    videos = resp.json().get("videos", [])
    if not videos:
        params["query"] = "abstract motion background"
        resp = requests.get("https://api.pexels.com/videos/search", headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        videos = resp.json().get("videos", [])
    if not videos:
        die(f"No Pexels results for '{keyword}' (or fallback query).")

    files = [f for f in videos[0]["video_files"] if f.get("file_type") == "video/mp4"]
    files.sort(key=lambda f: f.get("height", 0), reverse=True)
    link = files[0]["link"]

    clip_path = workdir / f"src_{idx:02d}.mp4"
    with requests.get(link, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(clip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
    return clip_path


def build_segment_video(clip_path, duration, idx, workdir):
    out_path = workdir / f"vid_{idx:02d}.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "-1", "-i", str(clip_path),
         "-t", f"{duration:.2f}",
         "-vf", f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}",
         "-an", "-r", "30", "-pix_fmt", "yuv420p", str(out_path)],
        check=True,
    )
    return out_path


def concat_media(paths, out_path, is_audio):
    list_path = out_path.with_suffix(".txt")
    list_path.write_text("\n".join(f"file '{p.resolve()}'" for p in paths))
    if is_audio:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
               "-i", str(list_path), "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2", str(out_path)]
    else:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
               "-i", str(list_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", str(out_path)]
    subprocess.run(cmd, check=True)
    return out_path


def mix_background_music(narration_path, duration, workdir):
    tracks = [p for p in MUSIC_DIR.glob("*") if p.suffix.lower() in (".mp3", ".wav", ".m4a")]
    if not tracks:
        return narration_path

    track = random.choice(tracks)
    mixed_path = workdir / "mixed_audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(narration_path.resolve()),
         "-stream_loop", "-1", "-t", f"{duration:.2f}", "-i", str(track.resolve()),
         "-filter_complex",
         f"[1:a]volume={MUSIC_VOLUME}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]",
         "-map", "[aout]", "-ar", "44100", "-ac", "2", str(mixed_path)],
        check=True,
    )
    return mixed_path


def render_caption_image(text, idx, workdir):
    # Homebrew's ffmpeg bottle ships without freetype/libass, so the drawtext
    # filter isn't available. Render captions as transparent PNGs with Pillow
    # instead and composite them with ffmpeg's overlay filter (no text engine
    # needed there).
    font = ImageFont.truetype(FONT_PATH, 66)
    lines = textwrap.wrap(text, width=18)

    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    stroke = 6
    line_height = max(draw.textbbox((0, 0), line, font=font, stroke_width=stroke)[3]
                      for line in lines) + 16
    y = int(H * 0.72)
    for line in lines:
        line_width = draw.textbbox((0, 0), line, font=font, stroke_width=stroke)[2]
        x = (W - line_width) / 2
        # Solid yellow fill with a thick black outline for legibility over footage.
        draw.text((x, y), line, font=font, fill=(*CAPTION_RGB, 255),
                  stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        y += line_height

    path = workdir / f"cap_{idx:02d}.png"
    img.save(path)
    return path


def render_watermark_image(workdir):
    # A persistent channel handle shown near the top of every frame.
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 42)
    tw = draw.textbbox((0, 0), BRAND_HANDLE, font=font)[2]
    draw.text(((W - tw) / 2, int(H * 0.06)), BRAND_HANDLE, font=font,
              fill=(255, 255, 255, 205), stroke_width=2, stroke_fill=(0, 0, 0, 190))
    path = workdir / "watermark.png"
    img.save(path)
    return path


def render_cta_overlay(workdir):
    # Branded follow call-to-action, overlaid ON the footage (not a separate
    # screen) during the closing narration.
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([70, int(H * 0.30), W - 70, int(H * 0.60)],
                           radius=32, fill=(0, 0, 0, 150))

    def center(text, size, y, fill, stroke=3):
        font = ImageFont.truetype(FONT_PATH, size)
        w = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)[2]
        draw.text(((W - w) / 2, y), text, font=font, fill=fill,
                  stroke_width=stroke, stroke_fill=(0, 0, 0, 255))

    center("FOLLOW", 118, int(H * 0.33), (255, 255, 255, 255), stroke=5)
    center(BRAND_HANDLE, 70, int(H * 0.455), (*ACCENT_RGB, 255), stroke=4)
    y = int(H * 0.545)
    for line in textwrap.wrap(BRAND_TAGLINE, width=26):
        center(line, 42, y, (235, 235, 235, 255), stroke=3)
        y += 54
    path = workdir / "cta_overlay.png"
    img.save(path)
    return path


def _chunk_text(text, max_words):
    words = text.split()
    chunks = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return chunks or [text]


def build_final_video(video_path, audio_path, segments, workdir, final_path):
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(video_path.resolve()), "-i", str(audio_path.resolve())]

    # Build overlay images with their time windows. Normal segments become
    # short, fast caption chunks; the outro segment gets a single follow CTA
    # overlaid on its footage (integrated, not a separate screen).
    overlays = []  # (image_path, start, end)
    cap_idx = 0
    for seg in segments:
        if seg.get("is_outro"):
            overlays.append((render_cta_overlay(workdir), seg["start"], seg["end"]))
            continue
        chunks = _chunk_text(seg["text"], CAPTION_WORDS_PER_CHUNK)
        span = (seg["end"] - seg["start"]) / len(chunks)
        for j, chunk in enumerate(chunks):
            overlays.append((render_caption_image(chunk, cap_idx, workdir),
                             seg["start"] + j * span, seg["start"] + (j + 1) * span))
            cap_idx += 1

    cap_inputs = []  # (ffmpeg_input_index, start, end)
    next_index = 2
    for image_path, start, end in overlays:
        cmd += ["-loop", "1", "-i", str(image_path.resolve())]
        cap_inputs.append((next_index, start, end))
        next_index += 1

    # Persistent watermark, overlaid on top for the whole video.
    watermark_path = render_watermark_image(workdir)
    cmd += ["-loop", "1", "-i", str(watermark_path.resolve())]
    watermark_index = next_index

    filters = []
    stream = "[0:v]"
    for input_index, start, end in cap_inputs:
        out_label = f"[c{input_index}]"
        filters.append(
            f"{stream}[{input_index}:v]overlay=0:0:"
            f"enable='between(t,{start:.2f},{end:.2f})'{out_label}"
        )
        stream = out_label
    filters.append(f"{stream}[{watermark_index}:v]overlay=0:0[vout]")

    cmd += ["-filter_complex", ";".join(filters),
            "-map", "[vout]", "-map", "1:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-shortest", str(final_path.resolve())]
    subprocess.run(cmd, check=True, cwd=str(workdir))


def upload_to_youtube(video_path, script):
    if not TOKEN_PATH.exists():
        die("token.json missing. Run 'python3 authorize_youtube.py' once to grant upload access.")

    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), YOUTUBE_SCOPES)
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            TOKEN_PATH.write_text(creds.to_json())
        except Exception:
            die("YouTube authorization expired and could not refresh. Run 'python3 authorize_youtube.py' again.")

    youtube = build("youtube", "v3", credentials=creds)
    body = {
        "snippet": {
            "title": script["title"],
            "description": script["description"],
            "tags": script["tags"],
            "categoryId": YOUTUBE_CATEGORY_ID,
        },
        "status": {
            "privacyStatus": "public",
            "selfDeclaredMadeForKids": False,
        },
    }
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response = request.execute()
    return f"https://youtube.com/watch?v={response['id']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", help="Force a specific topic instead of letting Claude pick one.")
    parser.add_argument("--no-upload", action="store_true", help="Generate the video but skip uploading to YouTube.")
    args = parser.parse_args()

    check_deps()

    print("-> Generating script with Claude...")
    script = generate_script(args.topic)
    print(f"   Topic: {script['topic']}")
    print(f"   Title: {script['title']}")

    slug = "".join(c if c.isalnum() else "-" for c in script["topic"].lower()).strip("-")[:40]
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{slug}-{uuid.uuid4().hex[:6]}"
    run_dir = OUTPUT_DIR / run_id
    workdir = run_dir / "tmp"
    workdir.mkdir(parents=True, exist_ok=True)

    audio_paths, video_paths, segments = [], [], []
    cursor = 0.0
    for i, seg in enumerate(script["segments"]):
        print(f"-> Segment {i+1}/{len(script['segments'])}: narrating...")
        wav_path, duration = synthesize_segment(seg["text"], i, workdir)
        audio_paths.append(wav_path)

        print(f"-> Segment {i+1}/{len(script['segments'])}: fetching footage for '{seg['keyword']}'...")
        clip_path = fetch_pexels_clip(seg["keyword"], i, workdir)
        video_paths.append(build_segment_video(clip_path, duration, i, workdir))

        segments.append({"text": seg["text"], "start": cursor, "end": cursor + duration})
        cursor += duration

    # Branded outro: spoken call-to-action over real footage (the follow CTA is
    # overlaid on the video rather than shown as a separate end screen).
    print("-> Adding branded outro...")
    tagline_lc = BRAND_TAGLINE[:1].lower() + BRAND_TAGLINE[1:]
    cta_text = f"Follow {BRAND_SPOKEN} for {tagline_lc}."
    outro_idx = len(script["segments"])
    wav_path, duration = synthesize_segment(cta_text, outro_idx, workdir)
    audio_paths.append(wav_path)
    clip_path = fetch_pexels_clip("cinematic aerial landscape", outro_idx, workdir)
    video_paths.append(build_segment_video(clip_path, duration, outro_idx, workdir))
    segments.append({"text": cta_text, "start": cursor, "end": cursor + duration, "is_outro": True})
    cursor += duration

    print("-> Concatenating audio and video...")
    full_audio = concat_media(audio_paths, workdir / "full_audio.wav", is_audio=True)
    full_video = concat_media(video_paths, workdir / "concatenated.mp4", is_audio=False)

    print("-> Mixing in background music...")
    full_audio = mix_background_music(full_audio, cursor, workdir)

    print("-> Burning in captions and rendering final video...")
    final_path = run_dir / "final.mp4"
    build_final_video(full_video, full_audio, segments, workdir, final_path)

    metadata = (
        f"TITLE:\n{script['title']}\n\n"
        f"DESCRIPTION:\n{script['description']}\n\n"
        f"TAGS:\n{', '.join(script['tags'])}\n"
    )
    (run_dir / "metadata.txt").write_text(metadata)

    save_used_topic(script["topic"])

    print(f"\nDone. Video: {final_path}")
    print(f"Metadata (title/description/tags): {run_dir / 'metadata.txt'}")

    if args.no_upload:
        print("Skipping upload (--no-upload).")
    else:
        print("-> Uploading to YouTube...")
        url = upload_to_youtube(final_path, script)
        print(f"Uploaded: {url}")


if __name__ == "__main__":
    main()
