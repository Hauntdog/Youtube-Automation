"""
YouTube Video Auto-Poster Bot with AI-Generated Metadata and Scheduler
Requires: pip install google-api-python-client google-auth-oauthlib google-auth-httplib2 google-generativeai
"""
import os
import json
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import threading

import google.generativeai as genai
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError


class Timer:
    """Simple timer class for tracking elapsed time."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.end_time = None
    
    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def formatted_elapsed(self) -> str:
        """Get formatted elapsed time string."""
        elapsed = self.elapsed()
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        elif elapsed < 3600:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            return f"{hours}h {minutes}m {seconds}s"


class ScheduledUpload:
    """Represents a scheduled video upload."""
    
    def __init__(self, video_path: str, scheduled_time: datetime, 
                 privacy: str = "private", tags: list = None, 
                 user_context: str = ""):
        self.video_path = video_path
        self.scheduled_time = scheduled_time
        self.privacy = privacy
        self.tags = tags or []
        self.user_context = user_context
        self.status = "pending"  # pending, uploading, completed, failed
    
    def time_until_upload(self) -> float:
        """Get seconds until scheduled upload time."""
        return (self.scheduled_time - datetime.now()).total_seconds()
    
    def formatted_time_until(self) -> str:
        """Get formatted time until upload."""
        seconds = self.time_until_upload()
        if seconds < 0:
            return "Now"
        
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            return f"{minutes}m"
        elif seconds < 86400:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
        else:
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            return f"{days}d {hours}h"


class YouTubeBot:
    """YouTube bot that uploads videos with AI-generated titles and descriptions."""
    
    SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
    MAX_TITLE_LENGTH = 100
    
    def __init__(self, gemini_api_key: str):
        """Initialize the bot with Gemini API key."""
        self.gemini_api_key = gemini_api_key
        self.youtube = None
        self.gemini_model = None
        self.timer = Timer()
        self.scheduled_uploads: List[ScheduledUpload] = []
        self.scheduler_running = False
        self.scheduler_thread = None
        
        # Get script directory
        self.script_dir = Path(__file__).parent.resolve()
        self.TOKEN_FILE = self.script_dir / 'token.pickle'
        self.CREDENTIALS_FILE = self.script_dir / 'client_secrets.json'
        
        self._setup_gemini()
    
    def _setup_gemini(self):
        """Configure Gemini AI."""
        try:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Gemini AI configured successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not configure Gemini AI: {e}")
            print("📝 Will use fallback metadata generation")
            self.gemini_model = None
    
    def authenticate(self) -> bool:
        """Authenticate with YouTube API using OAuth 2.0."""
        creds = None
        
        if self.TOKEN_FILE.exists():
            with open(self.TOKEN_FILE, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.CREDENTIALS_FILE.exists():
                    print(f"\n❌ Error: 'client_secrets.json' not found!")
                    print(f"📁 Looking in: {self.script_dir}")
                    print("\n📋 To get this file:")
                    print("1. Go to https://console.cloud.google.com/")
                    print("2. Create a project or select existing one")
                    print("3. Enable YouTube Data API v3")
                    print("4. Create OAuth 2.0 credentials (Desktop app)")
                    print("5. Download the JSON file and rename it to 'client_secrets.json'")
                    return False
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.CREDENTIALS_FILE), self.SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            with open(self.TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)
        
        self.youtube = build('youtube', 'v3', credentials=creds)
        return True
    
    def _sanitize_title(self, title: str) -> str:
        """Ensure title meets YouTube requirements."""
        title = title.strip()
        title = ' '.join(title.split())
        
        if len(title) > self.MAX_TITLE_LENGTH:
            title = title[:self.MAX_TITLE_LENGTH - 3] + "..."
        
        if not title:
            title = "Untitled Video"
        
        return title
    
    def _create_fallback_metadata(self, video_path: str, user_context: str = "") -> Dict[str, str]:
        """Create fallback metadata from filename."""
        video_name = Path(video_path).stem
        title = video_name
        
        import re
        title = re.sub(r'\[[\w-]+\]$', '', title)
        title = title.replace('｜｜', '-').replace('_', ' ')
        title = ' '.join(title.split())
        title = self._sanitize_title(title)
        
        description = f"Video: {video_name}"
        if user_context:
            description += f"\n\n{user_context}"
        
        return {
            "title": title,
            "description": description
        }
    
    def generate_metadata(self, video_path: str, user_context: str = "") -> Dict[str, str]:
        """Generate title and description using Gemini AI."""
        metadata_timer = Timer()
        metadata_timer.start()
        
        video_name = Path(video_path).stem
        
        if not self.gemini_model:
            result = self._create_fallback_metadata(video_path, user_context)
            metadata_timer.stop()
            print(f"⏱️  Metadata generated in {metadata_timer.formatted_elapsed()}")
            return result
        
        prompt = f"""Generate a YouTube video title and description for a video file named: "{video_name}"

{f"Additional context: {user_context}" if user_context else ""}

Please provide:
1. An engaging, SEO-friendly title (max 100 characters)
2. A detailed description (200-300 words) that includes:
   - What the video is about
   - Key points or highlights
   - Relevant hashtags (3-5)

Format your response as:
TITLE: [your title here]
DESCRIPTION: [your description here]"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            text = response.text
            
            title = ""
            description = ""
            
            if "TITLE:" in text and "DESCRIPTION:" in text:
                parts = text.split("DESCRIPTION:")
                title = parts[0].replace("TITLE:", "").strip()
                description = parts[1].strip()
            else:
                lines = text.strip().split('\n')
                title = lines[0].strip()
                description = '\n'.join(lines[1:]).strip()
            
            title = title.replace('**', '').replace('*', '').strip()
            title = self._sanitize_title(title)
            
            if not description:
                description = f"Video: {video_name}"
            
            metadata_timer.stop()
            print(f"⏱️  Metadata generated in {metadata_timer.formatted_elapsed()}")
            
            return {
                "title": title,
                "description": description
            }
        
        except Exception as e:
            print(f"⚠️  Error generating metadata with AI: {e}")
            print("📝 Using fallback metadata...")
            result = self._create_fallback_metadata(video_path, user_context)
            metadata_timer.stop()
            print(f"⏱️  Metadata generated in {metadata_timer.formatted_elapsed()}")
            return result
    
    def upload_video(
        self,
        video_path: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        category: str = "22",
        privacy: str = "private",
        tags: list = None,
        user_context: str = ""
    ) -> Optional[str]:
        """Upload a video to YouTube."""
        
        self.timer.start()
        
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file not found: {video_path}")
            return None
        
        if not title or not description:
            print("🤖 Generating title and description with AI...")
            metadata = self.generate_metadata(video_path, user_context)
            title = title or metadata["title"]
            description = description or metadata["description"]
        
        title = self._sanitize_title(title)
        
        print(f"\n📹 Uploading video: {Path(video_path).name}")
        print(f"📝 Title: {title}")
        print(f"📄 Description preview: {description[:100]}...")
        
        body = {
            'snippet': {
                'title': title,
                'description': description,
                'tags': tags or [],
                'categoryId': category
            },
            'status': {
                'privacyStatus': privacy,
                'selfDeclaredMadeForKids': False
            }
        }
        
        try:
            upload_timer = Timer()
            upload_timer.start()
            
            media = MediaFileUpload(
                video_path,
                chunksize=-1,
                resumable=True
            )
            
            request = self.youtube.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            response = None
            last_progress_time = time.time()
            
            while response is None:
                status, response = request.next_chunk()
                if status:
                    progress_pct = int(status.progress() * 100)
                    current_time = time.time()
                    
                    if current_time - last_progress_time >= 2 or progress_pct in [25, 50, 75, 100]:
                        elapsed = upload_timer.formatted_elapsed()
                        print(f"⬆️  Upload progress: {progress_pct}% | Elapsed: {elapsed}")
                        last_progress_time = current_time
            
            upload_timer.stop()
            self.timer.stop()
            
            video_id = response['id']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            print(f"\n✅ Upload successful!")
            print(f"🔗 Video URL: {video_url}")
            print(f"🆔 Video ID: {video_id}")
            print(f"⏱️  Upload time: {upload_timer.formatted_elapsed()}")
            print(f"⏱️  Total time: {self.timer.formatted_elapsed()}")
            
            return video_id
        
        except HttpError as e:
            self.timer.stop()
            print(f"❌ HTTP Error: {e.resp.status} - {e.content}")
            print(f"⏱️  Failed after: {self.timer.formatted_elapsed()}")
            return None
        except Exception as e:
            self.timer.stop()
            print(f"❌ Error uploading video: {e}")
            print(f"⏱️  Failed after: {self.timer.formatted_elapsed()}")
            return None
    
    def schedule_upload(self, upload: ScheduledUpload):
        """Add a video to the upload schedule."""
        self.scheduled_uploads.append(upload)
        print(f"✅ Scheduled upload for {upload.scheduled_time.strftime('%Y-%m-%d %I:%M %p')}")
        print(f"⏰ Time until upload: {upload.formatted_time_until()}")
    
    def start_scheduler(self):
        """Start the background scheduler thread."""
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        print("🚀 Scheduler started!")
    
    def stop_scheduler(self):
        """Stop the scheduler thread."""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        print("🛑 Scheduler stopped!")
    
    def _scheduler_loop(self):
        """Background loop that checks and executes scheduled uploads."""
        while self.scheduler_running:
            current_time = datetime.now()
            
            for upload in self.scheduled_uploads[:]:
                if upload.status == "pending" and current_time >= upload.scheduled_time:
                    print(f"\n{'='*60}")
                    print(f"⏰ SCHEDULED UPLOAD STARTING")
                    print(f"🕐 Scheduled for: {upload.scheduled_time.strftime('%Y-%m-%d %I:%M %p')}")
                    print(f"📹 Video: {Path(upload.video_path).name}")
                    print(f"{'='*60}\n")
                    
                    upload.status = "uploading"
                    
                    video_id = self.upload_video(
                        video_path=upload.video_path,
                        privacy=upload.privacy,
                        tags=upload.tags,
                        user_context=upload.user_context
                    )
                    
                    if video_id:
                        upload.status = "completed"
                        print(f"\n🎉 Scheduled upload completed successfully!")
                    else:
                        upload.status = "failed"
                        print(f"\n❌ Scheduled upload failed!")
                    
                    self.scheduled_uploads.remove(upload)
            
            time.sleep(30)  # Check every 30 seconds
    
    def list_scheduled_uploads(self):
        """Display all scheduled uploads."""
        if not self.scheduled_uploads:
            print("\n📭 No uploads scheduled")
            return
        
        print(f"\n{'='*60}")
        print("📅 SCHEDULED UPLOADS")
        print(f"{'='*60}")
        
        for i, upload in enumerate(self.scheduled_uploads, 1):
            print(f"\n{i}. {Path(upload.video_path).name}")
            print(f"   ⏰ Scheduled: {upload.scheduled_time.strftime('%Y-%m-%d %I:%M %p')}")
            print(f"   ⏳ Time until: {upload.formatted_time_until()}")
            print(f"   🔒 Privacy: {upload.privacy}")
            print(f"   📊 Status: {upload.status}")


def parse_schedule_time(time_str: str) -> Optional[datetime]:
    """Parse various time formats like '9am', '10:30pm', '14:00', '2025-01-15 9am'."""
    time_str = time_str.strip().lower()
    now = datetime.now()
    
    try:
        # Try full datetime format: "2025-01-15 9am" or "2025-01-15 14:00"
        if ' ' in time_str:
            date_part, time_part = time_str.split(' ', 1)
            date_obj = datetime.strptime(date_part, '%Y-%m-%d')
            
            # Parse time part
            if 'am' in time_part or 'pm' in time_part:
                time_part = time_part.replace('am', ' AM').replace('pm', ' PM')
                if ':' in time_part:
                    time_obj = datetime.strptime(time_part.strip(), '%I:%M %p')
                else:
                    time_obj = datetime.strptime(time_part.strip(), '%I %p')
            else:
                time_obj = datetime.strptime(time_part.strip(), '%H:%M')
            
            result = date_obj.replace(hour=time_obj.hour, minute=time_obj.minute, second=0)
            return result
        
        # Time only formats (assume today or tomorrow)
        if 'am' in time_str or 'pm' in time_str:
            time_str = time_str.replace('am', ' AM').replace('pm', ' PM')
            if ':' in time_str:
                time_obj = datetime.strptime(time_str.strip(), '%I:%M %p')
            else:
                time_obj = datetime.strptime(time_str.strip(), '%I %p')
        else:
            # 24-hour format
            if ':' in time_str:
                time_obj = datetime.strptime(time_str.strip(), '%H:%M')
            else:
                time_obj = datetime.strptime(time_str.strip(), '%H')
        
        # Combine with today's date
        result = now.replace(hour=time_obj.hour, minute=time_obj.minute, second=0, microsecond=0)
        
        # If time has passed today, schedule for tomorrow
        if result <= now:
            result += timedelta(days=1)
        
        return result
    
    except Exception as e:
        print(f"⚠️  Could not parse time: {e}")
        return None


def main():
    """Main function to run the YouTube bot."""
    print("=" * 60)
    print("🎬 YouTube Auto-Poster Bot with Scheduler")
    print("=" * 60)
    
    gemini_key = input("\n🔑 Enter your Gemini API key: ").strip()
    if not gemini_key:
        print("❌ Gemini API key is required!")
        return
    
    bot = YouTubeBot(gemini_api_key=gemini_key)
    
    print("\n🔐 Authenticating with YouTube...")
    if not bot.authenticate():
        return
    
    print("✅ Authentication successful!")
    
    session_timer = Timer()
    session_timer.start()
    total_uploads = 0
    successful_uploads = 0
    
    while True:
        print("\n" + "=" * 60)
        print("📤 MAIN MENU")
        print("=" * 60)
        print("1. Upload now")
        print("2. Schedule upload")
        print("3. View scheduled uploads")
        print("4. Quit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "4":
            break
        
        elif choice == "3":
            bot.list_scheduled_uploads()
            continue
        
        elif choice in ["1", "2"]:
            video_path = input("\n📁 Enter video file path: ").strip()
            video_path = video_path.strip('"').strip("'")
            
            if not os.path.exists(video_path):
                print(f"❌ File not found: {video_path}")
                continue
            
            context = input("\n💬 Enter context about the video (optional): ").strip()
            
            print("\n🔒 Privacy options: 1) Private  2) Unlisted  3) Public")
            privacy_choice = input("Select privacy (1-3, default: 1): ").strip() or "1"
            privacy_map = {"1": "private", "2": "unlisted", "3": "public"}
            privacy = privacy_map.get(privacy_choice, "private")
            
            tags_input = input("\n🏷️  Enter tags (comma-separated, optional): ").strip()
            tags = [tag.strip() for tag in tags_input.split(",")] if tags_input else []
            
            if choice == "1":
                # Upload now
                print("\n" + "-" * 60)
                total_uploads += 1
                video_id = bot.upload_video(
                    video_path=video_path,
                    privacy=privacy,
                    tags=tags,
                    user_context=context
                )
                
                if video_id:
                    successful_uploads += 1
                    print("\n🎉 VIDEO UPLOADED SUCCESSFULLY!")
            
            elif choice == "2":
                # Schedule upload
                print("\n⏰ Schedule upload time")
                print("Examples: '9am', '10:30pm', '14:00', '2025-01-15 9am'")
                time_str = input("Enter time: ").strip()
                
                scheduled_time = parse_schedule_time(time_str)
                if scheduled_time:
                    upload = ScheduledUpload(
                        video_path=video_path,
                        scheduled_time=scheduled_time,
                        privacy=privacy,
                        tags=tags,
                        user_context=context
                    )
                    bot.schedule_upload(upload)
                    
                    if not bot.scheduler_running:
                        bot.start_scheduler()
                else:
                    print("❌ Invalid time format")
        
        else:
            print("❌ Invalid option")
    
    bot.stop_scheduler()
    
    session_timer.stop()
    print("\n" + "=" * 60)
    print("📊 SESSION SUMMARY")
    print("=" * 60)
    print(f"⏱️  Total session time: {session_timer.formatted_elapsed()}")
    print(f"📤 Total uploads attempted: {total_uploads}")
    print(f"✅ Successful uploads: {successful_uploads}")
    if total_uploads > 0:
        success_rate = (successful_uploads / total_uploads) * 100
        print(f"📈 Success rate: {success_rate:.1f}%")
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()