import json
import logging
import os
import subprocess
import asyncio
import tempfile
import shutil
from base64 import b64decode
from pathlib import Path
from urllib.parse import urljoin
from typing import Optional
import requests
import glob
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
from telegram.constants import ParseMode, ChatAction
from telethon import TelegramClient
from telethon.errors import FloodWaitError
import time

# Load environment variables
load_dotenv()

# Configuration from environment variables
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH")
SESSION_NAME = os.getenv("SESSION_NAME", "vimeo_bot_session")
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", "0")) if os.getenv("ADMIN_USER_ID") else None

# Bot limits (convert GB to bytes)
FREE_USER_LIMIT = int(os.getenv("FREE_USER_LIMIT_GB", "2")) * 1024 * 1024 * 1024
PREMIUM_USER_LIMIT = int(os.getenv("PREMIUM_USER_LIMIT_GB", "4")) * 1024 * 1024 * 1024

# File paths
N_M3U8DL_RE_PATH = os.getenv("N_M3U8DL_RE_PATH", "./N_m3u8DL-RE")
TEMP_DIR_PREFIX = os.getenv("TEMP_DIR_PREFIX", "vimeo_bot_temp_")

# Validate required environment variables
if not BOT_TOKEN or not API_ID or not API_HASH:
    raise ValueError("Missing required environment variables: BOT_TOKEN, API_ID, API_HASH")

# Setup logging
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=log_level
)
logger = logging.getLogger(__name__)

# Global Telethon client
telethon_client = None

class VimeoDownloader:
    def __init__(self, playlist_url: str, output_path: str):
        self.playlist_url = playlist_url
        self.output_path = output_path
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        self.response_data = None
        self.clip_id = None
        self.videos = []
        self.audios = []
        self.main_base = None
        self.video_streams = []
        self.audio_streams = []

    def send_request(self) -> bool:
        logger.info('Sending request...')
        try:
            resp = requests.get(self.playlist_url, timeout=30)
            if resp.status_code != 200:
                return False
            self.response_data = json.loads(resp.text)
            return True
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.error(f"Request failed: {e}")
            return False

    def parse_playlist(self) -> bool:
        logger.info('Parsing playlist JSON...')
        try:
            parsed = self.response_data
            self.clip_id = parsed.get('clip_id')
            self.main_base = urljoin(self.playlist_url, parsed.get('base_url', ''))

            self.videos = sorted(parsed.get('video', []),
                                 key=lambda p: p.get('width', 1) * p.get('height', 1),
                                 reverse=True)
            self.audios = sorted(parsed.get('audio', []),
                                 key=lambda p: p.get('sample_rate', 1) * p.get('bitrate', 1),
                                 reverse=True)

            return bool(self.videos or self.audios)
        except Exception as e:
            logger.error(f"Parsing failed: {e}")
            return False

    def _save_playlist(self, stream: dict, content_type: str) -> tuple[str, str]:
        stream_base = urljoin(self.main_base, stream.get('base_url', ''))
        segments_to_write = []
        max_duration = 0

        for seg in stream.get('segments', []):
            duration = seg.get('end') - seg.get('start')
            if duration > max_duration:
                max_duration = duration
            segments_to_write.append({
                'url': urljoin(stream_base, seg.get('url')),
                'duration': duration
            })

        init_name = f"{stream.get('id', 'NO_ID')}_{content_type}_init.mp4"
        with open(os.path.join(self.output_path, init_name), 'wb') as f:
            f.write(b64decode(stream.get('init_segment')))

        playlist_name = f"{stream.get('id', 'NO_ID')}_{content_type}.m3u8"
        with open(os.path.join(self.output_path, playlist_name), 'w') as f:
            f.write("#EXTM3U\n")
            f.write("#EXT-X-VERSION:7\n")
            f.write("#EXT-X-MEDIA-SEQUENCE:0\n")
            f.write("#EXT-X-PLAYLIST-TYPE:VOD\n")
            f.write(f"#EXT-X-TARGETDURATION:{int(round(max_duration)) + 1}\n")
            f.write(f'#EXT-X-MAP:URI="{init_name}"\n')
            for seg in segments_to_write:
                f.write(f"#EXTINF:{seg['duration']},\n")
                f.write(f"{seg['url']}\n")
            f.write("#EXT-X-ENDLIST\n")

        return playlist_name, init_name

    def _save_video_stream(self, video: dict) -> dict:
        playlist, init = self._save_playlist(video, 'video')
        return {
            'url': playlist,
            'resolution': f"{video.get('width')}x{video.get('height')}",
            'bandwidth': video.get('bitrate'),
            'average_bandwidth': video.get('avg_bitrate'),
            'codecs': video.get('codecs'),
            'init': init
        }

    def _save_audio_stream(self, audio: dict) -> dict:
        playlist, init = self._save_playlist(audio, 'audio')
        return {
            'url': playlist,
            'channels': audio.get('channels'),
            'bitrate': audio.get('bitrate'),
            'sample_rate': audio.get('sample_rate'),
            'init': init
        }

    def _save_master(self, video_streams: list, audio_streams: list) -> str:
        master_name = f"master_{self.clip_id}.m3u8"
        with open(os.path.join(self.output_path, master_name), 'w') as f:
            f.write("#EXTM3U\n")
            f.write("#EXT-X-INDEPENDENT-SEGMENTS\n")
            stream_id = 0
            for a in audio_streams:
                f.write(
                    f'#EXT-X-MEDIA:TYPE=AUDIO,URI="{a["url"]}",GROUP-ID="default-audio-group",'
                    f'NAME="{a["bitrate"]/1000}_{a["sample_rate"]}_{stream_id}",CHANNELS="{a["channels"]}"\n'
                )
                stream_id += 1
            for v in video_streams:
                f.write(
                    f'#EXT-X-STREAM-INF:BANDWIDTH={v["bandwidth"]},AVERAGE-BANDWIDTH={v["average_bandwidth"]},'
                    f'CODECS="{v["codecs"]}",RESOLUTION={v["resolution"]},AUDIO="default-audio-group"\n'
                )
                f.write(f'{v["url"]}\n')
        return master_name

    def save_media(self) -> tuple[str, list]:
        self.video_streams = [self._save_video_stream(v) for v in self.videos]
        self.audio_streams = [self._save_audio_stream(a) for a in self.audios]
        master_file = self._save_master(self.video_streams, self.audio_streams)
        return master_file, [*self.video_streams, *self.audio_streams]


def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def convert_mkv_to_mp4(input_path: Path, output_path: Path) -> bool:
    """Convert MKV file to MP4 using FFmpeg."""
    try:
        cmd = ['ffmpeg', '-i', str(input_path), '-c', 'copy', '-y', str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        return False


async def get_user_info(user_id: int):
    """Get user information to check if they have premium."""
    try:
        if telethon_client and telethon_client.is_connected():
            user = await telethon_client.get_entity(user_id)
            return user.premium if hasattr(user, 'premium') else False
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
    return False


def get_file_size_limit(is_premium: bool) -> int:
    """Get file size limit based on user premium status."""
    return PREMIUM_USER_LIMIT if is_premium else FREE_USER_LIMIT


def format_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"


async def start(update: Update, context: CallbackContext):
    """Start command handler."""
    welcome_text = """
🎥 **Vimeo Downloader Bot**

Send me a Vimeo playlist.json URL and I'll download and convert it to MP4 for you!

**Features:**
✅ Downloads Vimeo videos
✅ Converts MKV to MP4 automatically
✅ Supports large file uploads (up to 4GB for Premium users)
✅ Smart file size detection

**Usage:**
Just send me the Vimeo playlist.json URL and I'll handle the rest!

**File Limits:**
• Free users: Up to 2GB
• Premium users: Up to 4GB
    """
    
    keyboard = [
        [InlineKeyboardButton("ℹ️ Help", callback_data="help"),
         InlineKeyboardButton("📊 Status", callback_data="status")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        welcome_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )


async def help_command(update: Update, context: CallbackContext):
    """Help command handler."""
    help_text = """
🆘 **How to use this bot:**

1️⃣ Copy your Vimeo playlist.json URL
2️⃣ Send it to me
3️⃣ Wait for the download and conversion
4️⃣ Receive your MP4 file!

**Supported URLs:**
• Vimeo playlist.json URLs
• Must be valid and accessible

**Requirements:**
• FFmpeg must be installed on the server
• Valid Vimeo playlist URL

**Limits:**
• Free users: 2GB max file size
• Premium users: 4GB max file size

If you encounter any issues, contact the bot administrator.
    """
    
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def admin_command(update: Update, context: CallbackContext):
    """Admin only command to check bot status."""
    if ADMIN_USER_ID and update.effective_user.id != ADMIN_USER_ID:
        await update.message.reply_text("❌ You don't have permission to use this command.")
        return
    
    status_text = "🔧 **Admin Status Panel:**\n\n"
    status_text += f"🤖 Bot Token: {'✅ Set' if BOT_TOKEN else '❌ Missing'}\n"
    status_text += f"🔑 API ID: {'✅ Set' if API_ID else '❌ Missing'}\n"
    status_text += f"🔒 API Hash: {'✅ Set' if API_HASH else '❌ Missing'}\n"
    status_text += f"🛠️ FFmpeg: {'✅ Available' if check_ffmpeg() else '❌ Missing'}\n"
    status_text += f"📡 Telethon: {'✅ Connected' if telethon_client and telethon_client.is_connected() else '❌ Disconnected'}\n"
    status_text += f"📁 N_m3u8DL-RE: {'✅ Found' if os.path.exists(N_M3U8DL_RE_PATH) else '❌ Missing'}\n\n"
    status_text += f"📊 **Limits:**\n"
    status_text += f"• Free users: {format_size(FREE_USER_LIMIT)}\n"
    status_text += f"• Premium users: {format_size(PREMIUM_USER_LIMIT)}\n"
    
    await update.message.reply_text(status_text, parse_mode=ParseMode.MARKDOWN)
    """Handle inline keyboard button presses."""
    query = update.callback_query
    await query.answer()
    
    if query.data == "help":
        await help_command(query, context)
    elif query.data == "status":
        status_text = "🤖 **Bot Status:**\n\n"
        status_text += f"✅ Bot is running\n"
        status_text += f"{'✅' if check_ffmpeg() else '❌'} FFmpeg available\n"
        status_text += f"{'✅' if telethon_client and telethon_client.is_connected() else '❌'} Telethon connected\n"
        
        await query.edit_message_text(status_text, parse_mode=ParseMode.MARKDOWN)


async def process_vimeo_url(update: Update, context: CallbackContext):
    """Process Vimeo playlist URL."""
    url = update.message.text.strip()
    user_id = update.effective_user.id
    
    # Validate URL
    if not url.startswith('https://') or 'playlist.json' not in url:
        await update.message.reply_text(
            "❌ Invalid URL! Please send a valid Vimeo playlist.json URL."
        )
        return
    
    # Check if user has premium
    is_premium = await get_user_info(user_id)
    file_limit = get_file_size_limit(is_premium)
    
    status_msg = await update.message.reply_text(
        f"🔄 Processing your request...\n"
        f"👤 User: {'Premium' if is_premium else 'Free'}\n"
        f"📏 File limit: {format_size(file_limit)}"
    )
    
    # Create temporary directory with custom prefix
    temp_dir = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)
    try:
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
            
            # Download and process
            downloader = VimeoDownloader(url, temp_dir)
            
            await status_msg.edit_text("🔄 Fetching playlist information...")
            
            if not downloader.send_request():
                await status_msg.edit_text("❌ Failed to fetch playlist. Check your URL.")
                return
                
            if not downloader.parse_playlist():
                await status_msg.edit_text("❌ Failed to parse playlist.")
                return
            
            await status_msg.edit_text("🔄 Creating download playlists...")
            master_file, streams = downloader.save_media()
            
            await status_msg.edit_text("🔄 Starting download...")
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_DOCUMENT)
            
            # Download using N_m3u8DL-RE
            try:
                subprocess.run([
                    N_M3U8DL_RE_PATH,
                    os.path.join(temp_dir, master_file),
                    "-M", "format=mkv",
                    "--workDir", temp_dir
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                await status_msg.edit_text(f"❌ Download failed: {e}")
                return
            
            # Find downloaded MKV file
            mkv_files = glob.glob(os.path.join(temp_dir, "*.mkv"))
            if not mkv_files:
                await status_msg.edit_text("❌ No MKV file found after download.")
                return
            
            mkv_path = Path(mkv_files[0])
            mp4_path = mkv_path.with_suffix('.mp4')
            
            # Check file size before conversion
            file_size = mkv_path.stat().st_size
            if file_size > file_limit:
                await status_msg.edit_text(
                    f"❌ File too large!\n"
                    f"File size: {format_size(file_size)}\n"
                    f"Your limit: {format_size(file_limit)}\n"
                    f"{'Consider upgrading to Premium!' if not is_premium else 'File exceeds Premium limit!'}"
                )
                return
            
            await status_msg.edit_text(f"🔄 Converting to MP4... ({format_size(file_size)})")
            
            # Convert MKV to MP4
            if check_ffmpeg():
                if convert_mkv_to_mp4(mkv_path, mp4_path):
                    # Remove original MKV file
                    mkv_path.unlink()
                    final_file = mp4_path
                else:
                    await status_msg.edit_text("⚠️ Conversion failed, uploading MKV file...")
                    final_file = mkv_path
            else:
                await status_msg.edit_text("⚠️ FFmpeg not available, uploading MKV file...")
                final_file = mkv_path
            
            await status_msg.edit_text("🔄 Uploading file...")
            
            # Upload file
            final_size = final_file.stat().st_size
            
            # Use Telethon for files > 50MB, regular bot API for smaller files
            if final_size > 50 * 1024 * 1024 and telethon_client and telethon_client.is_connected():
                try:
                    await status_msg.edit_text("🔄 Uploading large file via Telethon...")
                    await telethon_client.send_file(
                        update.effective_chat.id,
                        final_file,
                        caption=f"📹 Video downloaded and converted!\n"
                                f"📊 Size: {format_size(final_size)}\n"
                                f"👤 User: {'Premium' if is_premium else 'Free'}",
                        progress_callback=lambda current, total: None
                    )
                except Exception as e:
                    logger.error(f"Telethon upload failed: {e}")
                    await status_msg.edit_text("❌ Upload failed via Telethon. File might be too large.")
                    return
            else:
                # Use regular bot API
                try:
                    with open(final_file, 'rb') as f:
                        await context.bot.send_document(
                            chat_id=update.effective_chat.id,
                            document=f,
                            caption=f"📹 Video downloaded and converted!\n"
                                    f"📊 Size: {format_size(final_size)}\n"
                                    f"👤 User: {'Premium' if is_premium else 'Free'}",
                            filename=final_file.name
                        )
                except Exception as e:
                    logger.error(f"Bot API upload failed: {e}")
                    await status_msg.edit_text("❌ Upload failed. File might be too large for bot API.")
                    return
            
            await status_msg.edit_text("✅ Complete! File uploaded successfully.")
            
            # Cleanup temp files
            for stream in streams:
                try:
                    os.remove(os.path.join(temp_dir, stream['url']))
                    os.remove(os.path.join(temp_dir, stream['init']))
                except FileNotFoundError:
                    pass
            try:
                os.remove(os.path.join(temp_dir, master_file))
            except FileNotFoundError:
                pass
                
        except Exception as e:
            logger.error(f"Processing error: {e}")
            await status_msg.edit_text(f"❌ An error occurred: {str(e)}")
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            logger.error(f"Failed to cleanup temp directory: {e}")


async def initialize_telethon():
    """Initialize Telethon client."""
    global telethon_client
    try:
        telethon_client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
        await telethon_client.start()
        logger.info("Telethon client started successfully")
    except Exception as e:
        logger.error(f"Failed to start Telethon client: {e}")
        telethon_client = None


def main():
    """Main function to run the bot."""
    # Check FFmpeg
    if not check_ffmpeg():
        logger.warning("FFmpeg not found! Install FFmpeg for MKV to MP4 conversion.")
    
    # Create application
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("admin", admin_command))
    application.add_handler(CallbackQueryHandler(button_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_vimeo_url))
    
    # Initialize Telethon
    async def post_init(app):
        await initialize_telethon()
    
    application.post_init = post_init
    
    # Run the bot
    logger.info("Starting bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()