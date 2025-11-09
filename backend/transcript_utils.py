import re
import os
import json
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

def extract_video_id(url: str) -> str:
    """Extracts YouTube Video ID from any YouTube URL."""
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None


def fetch_transcript(video_id: str) -> str:
    """Fetch transcript with caching. Returns None if not available."""
    os.makedirs("cache", exist_ok=True)
    cache_path = f"cache/{video_id}.json"

    # ✅ Return cached transcript if exists
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f).get("transcript")

    # ✅ Else fetch from YouTube
    try:
        transcript_raw = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join(item["text"] for item in transcript_raw)

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"transcript": transcript_text}, f)

        return transcript_text

    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None
