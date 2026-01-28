import os
import argparse
import yt_dlp
from pathlib import Path

def download_youtube_videos(keywords, output_dir, max_videos=5):
    """
    Download videos from YouTube based on keywords.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': str(output_path / '%(id)s.%(ext)s'),
        'noplaylist': True,
        'quiet': False,
        'max_downloads': max_videos,
        'ignoreerrors': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for keyword in keywords:
            print(f"Searching and downloading for: {keyword}")
            try:
                # 'ytsearchN:KEYWORD' searches for N videos
                ydl.download([f"ytsearch{max_videos}:{keyword}"])
            except Exception as e:
                print(f"Error downloading {keyword}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keywords', type=str, nargs='+', required=True, help='List of search keywords')
    parser.add_argument('--output_dir', type=str, default='data/raw/youtube', help='Output directory')
    parser.add_argument('--max', type=int, default=5, help='Max videos per keyword')
    args = parser.parse_args()
    
    download_youtube_videos(args.keywords, args.output_dir, args.max)
