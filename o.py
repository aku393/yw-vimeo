#!/usr/bin/env python3
# o.py - Updated and fixed version (progress updates deduplicated & rate-limited)

import json
import logging
import os
import subprocess
import asyncio
import tempfile
import shutil
import time
import glob
import re
import threading
import queue
import psutil
import platform
import speedtest
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
from base64 import b64decode
from pathlib import Path
from urllib.parse import urljoin
from typing import Optional, Union, Dict, List, Tuple
import concurrent.futures
from dataclasses import dataclass, field
import aiofiles
import aiohttp
import requests
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Message, CallbackQuery
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
from telegram.constants import ParseMode, ChatAction
import nest_asyncio

# Load environment variables
load_dotenv()

# Enhanced Configuration from environment variables
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH")
SESSION_NAME = os.getenv("SESSION_NAME", "vimeo_bot_session")
ADMIN_USER_ID = int(os.getenv("ADMIN_USER_ID", "0")) if os.getenv("ADMIN_USER_ID") else None

# Bot limits (convert GB to bytes)
FREE_USER_LIMIT = int(os.getenv("FREE_USER_LIMIT_GB", "2")) * 1024 * 1024 * 1024
PREMIUM_USER_LIMIT = int(os.getenv("PREMIUM_LIMIT_GB", "4")) * 1024 * 1024 * 1024

# File paths
N_M3U8DL_RE_PATH = os.getenv("N_M3U8DL_RE_PATH", "./N_m3u8DL-RE")
TEMP_DIR_PREFIX = os.getenv("TEMP_DIR_PREFIX", "vimeo_bot_temp_")

# Enhanced Configuration
MAX_CONCURRENT_DOWNLOADS = int(os.getenv("MAX_CONCURRENT_DOWNLOADS", "5"))
PROGRESS_UPDATE_INTERVAL = float(os.getenv("PROGRESS_UPDATE_INTERVAL", "3.0"))  # seconds between edits
MIN_PROGRESS_DELTA = float(os.getenv("MIN_PROGRESS_DELTA", "0.01"))  # 1% change required to edit
DB_PATH = os.getenv("DB_PATH", "vimeo_bot.db")
ANALYTICS_ENABLED = os.getenv("ANALYTICS_ENABLED", "true").lower() == "true"
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "100"))

@dataclass
class DownloadStats:
    """Statistics for download operations"""
    start_time: float = field(default_factory=time.time)
    bytes_downloaded: int = 0
    total_bytes: int = 0
    speed_mbps: float = 0.0
    eta_seconds: int = 0
    stage: str = "Initializing"
    error_count: int = 0
    last_update: float = field(default_factory=time.time)
    current_segment: int = 0
    total_segments: int = 0
    peak_speed: float = 0.0

@dataclass
class UserSession:
    """User session data"""
    user_id: int
    username: str
    is_premium: bool
    downloads_today: int = 0
    total_downloads: int = 0
    bytes_downloaded_today: int = 0
    last_active: datetime = field(default_factory=datetime.now)
    current_downloads: int = 0
    favorite_quality: str = "highest"
    notification_preferences: Dict = field(default_factory=dict)

class PerformanceMonitor:
    """System performance monitoring with advanced metrics"""
    def __init__(self):
        self.start_time = time.time()
        self.stats_history = []
        self.max_history = 100
        self.peak_concurrent = 0
        self.total_processed = 0
        self.error_count = 0
        
    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        current_process = psutil.Process()
        process_memory = current_process.memory_info()
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_free_gb': disk.free / (1024**3),
            'network_sent_mb': network.bytes_sent / (1024**2),
            'network_recv_mb': network.bytes_recv / (1024**2),
            'uptime_hours': (time.time() - self.start_time) / 3600,
            'process_memory_mb': process_memory.rss / (1024**2),
            'open_files': len(current_process.open_files()),
            'threads': current_process.num_threads(),
            'connections': len(current_process.connections()) if hasattr(current_process, 'connections') else 0
        }
    
    def add_stats(self):
        """Add current stats to history"""
        stats = self.get_system_stats()
        stats['timestamp'] = time.time()
        stats['active_downloads'] = len(active_downloads)
        self.stats_history.append(stats)
        
        if stats['active_downloads'] > self.peak_concurrent:
            self.peak_concurrent = stats['active_downloads']
        
        if len(self.stats_history) > self.max_history:
            self.stats_history.pop(0)
    
    def get_average_stats(self, minutes: int = 5) -> Dict:
        """Get average statistics over the last N minutes"""
        cutoff_time = time.time() - (minutes * 60)
        recent_stats = [s for s in self.stats_history if s['timestamp'] > cutoff_time]
        
        if not recent_stats:
            return self.get_system_stats()
        
        avg_stats = {}
        for key in recent_stats[0].keys():
            if key != 'timestamp' and isinstance(recent_stats[0][key], (int, float)):
                avg_stats[key] = sum(s[key] for s in recent_stats) / len(recent_stats)
        
        return avg_stats

class DatabaseManager:
    """Enhanced database management with advanced analytics"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with enhanced schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('PRAGMA journal_mode=WAL;')
            conn.execute('''CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    username TEXT,
                    first_name TEXT,
                    last_name TEXT,
                    is_premium BOOLEAN DEFAULT FALSE,
                    total_downloads INTEGER DEFAULT 0,
                    total_bytes_downloaded INTEGER DEFAULT 0,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    preferred_quality TEXT DEFAULT 'highest',
                    language_code TEXT,
                    timezone TEXT,
                    banned BOOLEAN DEFAULT FALSE,
                    ban_reason TEXT
                )''')
            
            conn.execute('''CREATE TABLE IF NOT EXISTS downloads (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    url TEXT,
                    title TEXT,
                    file_size INTEGER,
                    download_speed REAL,
                    conversion_time REAL,
                    quality TEXT,
                    format TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    status TEXT,
                    error_message TEXT,
                    server_location TEXT,
                    user_ip TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )''')
            
            conn.execute('''CREATE TABLE IF NOT EXISTS system_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    cpu_percent REAL,
                    memory_percent REAL,
                    disk_free_gb REAL,
                    active_downloads INTEGER,
                    total_downloads_today INTEGER,
                    network_sent_mb REAL,
                    network_recv_mb REAL,
                    error_count INTEGER DEFAULT 0
                )''')
            
            conn.execute('''CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id INTEGER,
                    error_type TEXT,
                    error_message TEXT,
                    traceback TEXT,
                    url TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )''')
            
            conn.execute('''CREATE TABLE IF NOT EXISTS bot_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
    
    def update_user(self, user_id: int, username: str, first_name: str, 
                   last_name: str, is_premium: bool, language_code: str = None):
        """Update or insert user information"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''INSERT OR REPLACE INTO users 
                (user_id, username, first_name, last_name, is_premium, language_code, last_seen) 
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)''',
                         (user_id, username, first_name, last_name, is_premium, language_code))
    
    def log_download(self, user_id: int, url: str, title: str, file_size: int, 
                    download_speed: float, quality: str, format_type: str, 
                    status: str, error_message: str = None, conversion_time: float = 0):
        """Log download statistics with enhanced data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''INSERT INTO downloads 
                (user_id, url, title, file_size, download_speed, conversion_time, 
                 quality, format, start_time, end_time, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now', '-1 hour'), CURRENT_TIMESTAMP, ?, ?)''',
                         (user_id, url, title, file_size, download_speed, conversion_time, 
                          quality, format_type, status, error_message))
    
    def log_error(self, user_id: int, error_type: str, error_message: str, 
                  traceback: str = None, url: str = None):
        """Log error information"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''INSERT INTO error_logs 
                (user_id, error_type, error_message, traceback, url)
                VALUES (?, ?, ?, ?, ?)''', (user_id, error_type, error_message, traceback, url))
    
    def get_user_stats(self, user_id: int) -> Dict:
        """Get comprehensive user statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''SELECT 
                    COUNT(*) as total_downloads,
                    COALESCE(SUM(file_size), 0) as total_bytes,
                    COALESCE(AVG(download_speed), 0) as avg_speed,
                    COALESCE(MAX(download_speed), 0) as max_speed,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_downloads,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_downloads,
                    COUNT(CASE WHEN date(start_time) = date('now') THEN 1 END) as downloads_today,
                    COALESCE(SUM(CASE WHEN date(start_time) = date('now') THEN file_size ELSE 0 END), 0) as bytes_today
                FROM downloads WHERE user_id = ?''', (user_id,))
            
            result = cursor.fetchone()
            if result is None:
                return {
                    'total_downloads': 0,
                    'total_bytes': 0,
                    'avg_speed': 0.0,
                    'max_speed': 0.0,
                    'successful_downloads': 0,
                    'failed_downloads': 0,
                    'downloads_today': 0,
                    'bytes_today': 0
                }
            stats = dict(zip([col[0] for col in cursor.description], result))
            
            total = stats.get('total_downloads', 0)
            if total > 0:
                stats['success_rate'] = (stats.get('successful_downloads', 0) / total) * 100
            else:
                stats['success_rate'] = 0
                
            return stats
    
    def get_global_stats(self) -> Dict:
        """Get comprehensive global bot statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''SELECT 
                    COUNT(DISTINCT user_id) as total_users,
                    COUNT(*) as total_downloads,
                    COALESCE(SUM(file_size), 0) as total_bytes_served,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as successful_downloads,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_downloads,
                    COALESCE(AVG(download_speed), 0) as avg_download_speed,
                    COUNT(CASE WHEN date(start_time) = date('now') THEN 1 END) as downloads_today,
                    COUNT(CASE WHEN datetime(start_time) > datetime('now', '-1 hour') THEN 1 END) as downloads_last_hour
                FROM downloads''')
            
            result = cursor.fetchone()
            if result is None:
                stats = {}
            else:
                stats = dict(zip([col[0] for col in cursor.description], result))
            
            cursor2 = conn.execute('SELECT COUNT(*) FROM users WHERE is_premium = TRUE')
            premium_count_row = cursor2.fetchone()
            stats['premium_users'] = premium_count_row[0] if premium_count_row else 0
            
            total = stats.get('total_downloads', 0)
            if total > 0:
                stats['success_rate'] = (stats.get('successful_downloads', 0) / total) * 100
            else:
                stats['success_rate'] = 0
                
            return stats
    
    def get_top_users(self, limit: int = 10) -> List[Tuple]:
        """Get top users by download count and data usage"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''SELECT u.username, u.first_name, u.is_premium,
                       COUNT(d.id) as downloads, 
                       COALESCE(SUM(d.file_size), 0) as total_bytes,
                       COALESCE(AVG(d.download_speed), 0) as avg_speed
                FROM users u
                LEFT JOIN downloads d ON u.user_id = d.user_id AND d.status = 'completed'
                GROUP BY u.user_id
                HAVING downloads > 0
                ORDER BY downloads DESC, total_bytes DESC
                LIMIT ?''', (limit,))
            return cursor.fetchall()
    
    def get_download_trends(self, days: int = 7) -> Dict:
        """Get download trends over the last N days"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''SELECT 
                    date(start_time) as download_date,
                    COUNT(*) as daily_downloads,
                    COUNT(DISTINCT user_id) as daily_users,
                    COALESCE(SUM(file_size), 0) as daily_bytes
                FROM downloads
                WHERE start_time >= date('now', '-{} days')
                GROUP BY date(start_time)
                ORDER BY download_date'''.format(days))
            
            trends = []
            for row in cursor.fetchall():
                trends.append({
                    'date': row[0],
                    'downloads': row[1],
                    'users': row[2],
                    'bytes': row[3]
                })
            
            return trends

# Global instances
performance_monitor = PerformanceMonitor()
db_manager = DatabaseManager(DB_PATH)
active_downloads: Dict[int, DownloadStats] = {}
user_sessions: Dict[int, UserSession] = {}
download_queue = asyncio.Queue(maxsize=MAX_CONCURRENT_DOWNLOADS * 2)
progress_tasks: Dict[int, asyncio.Task] = {}
session_downloaders: Dict[int, "EnhancedVimeoDownloader"] = {}  # store active downloader instances
progress_message_ids: Dict[int, Tuple[int, int]] = {}  # user_id -> (chat_id, message_id)

def format_size(size_bytes: int) -> str:
    """Format file size in human readable format with more precision"""
    if not isinstance(size_bytes, (int, float)):
        return "0 B"
    size_bytes = int(size_bytes)
    if size_bytes == 0:
        return "0 B"
    elif size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.2f} GB"

def format_speed(bytes_per_sec: float) -> str:
    """Format download speed with dynamic units"""
    if bytes_per_sec is None:
        return "0 B/s"
    if bytes_per_sec == 0:
        return "0 B/s"
    bytes_per_sec = float(bytes_per_sec)
    if bytes_per_sec < 1024:
        return f"{bytes_per_sec:.1f} B/s"
    elif bytes_per_sec < 1024**2:
        return f"{bytes_per_sec/1024:.1f} KB/s"
    elif bytes_per_sec < 1024**3:
        return f"{bytes_per_sec/(1024**2):.2f} MB/s"
    else:
        return f"{bytes_per_sec/(1024**3):.3f} GB/s"

def format_time(seconds: int) -> str:
    """Format time duration with better precision"""
    try:
        seconds = int(seconds)
    except Exception:
        return "0s"
    if seconds <= 0:
        return "0s"
    elif seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds}s" if remaining_seconds > 0 else f"{minutes}m"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"

def create_progress_bar(progress: float, length: int = 25, style: str = "blocks") -> str:
    """Create enhanced visual progress bar with different styles"""
    progress = max(0, min(1, progress))
    filled = int(length * progress)
    
    if style == "blocks":
        full_block = "█"
        empty_block = "░"
        bar = full_block * filled + empty_block * (length - filled)
    elif style == "arrows":
        full_block = "▶"
        empty_block = "▷"
        bar = full_block * filled + empty_block * (length - filled)
    elif style == "circles":
        full_block = "●"
        empty_block = "○"
        bar = full_block * filled + empty_block * (length - filled)
    else:
        full_block = "█"
        empty_block = "░"
        bar = full_block * filled + empty_block * (length - filled)
    
    percentage = f"{progress*100:.1f}%"
    return f"{bar} {percentage}"

def escape_markdown(text: str) -> str:
    """Enhanced markdown escaping for Telegram MarkdownV2"""
    if not text:
        return ""
    # escape Telegram MarkdownV2 special characters
    escape_chars = r'([_*[\]()~`>#+\-=|{}.!\\])'
    escaped = re.sub(escape_chars, r'\\\1', str(text))
    return escaped

def generate_random_tips() -> str:
    """Generate random tips for users"""
    tips = [
        "💡 Tip: Premium users get faster upload speeds and larger file limits!",
        "🔥 Tip: Use /speedtest to check your connection before large downloads!",
        "⚡ Tip: The bot automatically retries failed downloads up to 3 times!",
        "🎯 Tip: Higher quality videos take longer to download but are worth it!",
        "📊 Tip: Check /stats to see your download history and achievements!",
        "🏆 Tip: Compete with other users on the /leaderboard!",
        "💾 Tip: Files are automatically cleaned up after successful uploads!",
        "🔄 Tip: The bot converts MKV to MP4 for better compatibility!",
        "📱 Tip: You can download up to {} concurrent videos!".format(MAX_CONCURRENT_DOWNLOADS),
        "🌟 Tip: Premium detection is automatic - no setup required!"
    ]
    return tips[int(time.time()) % len(tips)]

async def send_markdown_message(obj: Union[Update, Message, CallbackQuery], text: str, 
                               reply_markup=None, parse_mode=ParseMode.MARKDOWN_V2) -> Optional[Message]:
    """Enhanced message sending with comprehensive error handling and fallbacks.
       This function will try to edit messages where appropriate to avoid duplicates."""
    if not text:
        return None
        
    escaped_text = escape_markdown(text) if parse_mode == ParseMode.MARKDOWN_V2 else text
    
    # Telegram has a ~4096 char limit; trim if necessary
    if len(escaped_text) > 3900:
        escaped_text = escaped_text[:3890] + "\n\n\\.\\.\\. \\(message truncated\\)"
    
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            # If Update with callback_query: try to edit callback's message first
            if isinstance(obj, Update):
                if obj.callback_query:
                    try:
                        return await obj.callback_query.edit_message_text(
                            escaped_text, parse_mode=parse_mode, reply_markup=reply_markup
                        )
                    except Exception:
                        if obj.message:
                            return await obj.message.reply_text(
                                escaped_text, parse_mode=parse_mode, reply_markup=reply_markup
                            )
                elif obj.message:
                    return await obj.message.reply_text(
                        escaped_text, parse_mode=parse_mode, reply_markup=reply_markup
                    )

            elif isinstance(obj, CallbackQuery):
                try:
                    return await obj.edit_message_text(
                        escaped_text, parse_mode=parse_mode, reply_markup=reply_markup
                    )
                except Exception:
                    if obj.message:
                        return await obj.message.reply_text(
                            escaped_text, parse_mode=parse_mode, reply_markup=reply_markup
                        )

            elif isinstance(obj, Message):
                # Try to edit if it's an existing message (if allowed), else reply
                try:
                    return await obj.edit_text(
                        escaped_text, parse_mode=parse_mode, reply_markup=reply_markup
                    )
                except Exception:
                    return await obj.reply_text(
                        escaped_text, parse_mode=parse_mode, reply_markup=reply_markup
                    )
            else:
                # fallback: we don't have a message object - nothing to do
                return None
            
        except Exception as e:
            logger.error(f"Message send attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                try:
                    # Final fallback: send plain text without special markdown
                    plain_text = re.sub(r'[*_`\[\]()~>#+=|{}.!\\-]', '', text)
                    if isinstance(obj, Update) and obj.message:
                        return await obj.message.reply_text(plain_text)
                    elif isinstance(obj, Message):
                        return await obj.reply_text(plain_text)
                    elif isinstance(obj, CallbackQuery) and obj.message:
                        return await obj.message.reply_text(plain_text)
                except Exception as final_error:
                    logger.error(f"All message send attempts failed: {final_error}")
    return None

# Validate required environment variables
if not BOT_TOKEN or not API_ID or not API_HASH:
    print("❌ Error: Missing required environment variables!")
    print("Please check your .env file and ensure these variables are set:")
    print("- BOT_TOKEN")
    print("- API_ID") 
    print("- API_HASH")
    exit(1)

# Display configuration
print("🚀" + "="*60)
print("🔥 INSANE VIMEO DOWNLOADER BOT v2.0 - INITIALIZATION")
print("="*62)
print(f"✅ Configuration loaded successfully!")
print(f"📊 Free user limit: {format_size(FREE_USER_LIMIT)}")
print(f"📊 Premium user limit: {format_size(PREMIUM_USER_LIMIT)}")
print(f"📁 N_m3u8DL-RE path: {N_M3U8DL_RE_PATH}")
print(f"🔧 Max concurrent downloads: {MAX_CONCURRENT_DOWNLOADS}")
print(f"⚡ Progress update interval: {PROGRESS_UPDATE_INTERVAL}s")
print(f"💾 Database path: {DB_PATH}")
print(f"📊 Analytics: {'✅ Enabled' if ANALYTICS_ENABLED else '❌ Disabled'}")
print(f"🎯 Admin user ID: {ADMIN_USER_ID}")
print("="*62)

# Enhanced logging setup
log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper())
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=log_level,
    handlers=[
        logging.FileHandler('vimeo_bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

perf_logger = logging.getLogger('performance')
perf_handler = logging.FileHandler('performance.log', encoding='utf-8')
perf_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
perf_logger.addHandler(perf_handler)
perf_logger.setLevel(logging.INFO)

# Global Telethon client
telethon_client = None

class EnhancedVimeoDownloader:
    """Enhanced Vimeo downloader with advanced progress tracking and optimization"""
    
    def __init__(self, playlist_url: str, output_path: str, user_id: int):
        self.playlist_url = playlist_url
        self.output_path = output_path
        self.user_id = user_id
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        self.response_data = None
        self.clip_id = None
        self.videos = []
        self.audios = []
        self.main_base = None
        self.video_streams = []
        self.audio_streams = []
        
        self.stats = DownloadStats()
        self.stats.stage = "Initializing"
        active_downloads[user_id] = self.stats
        
        self.max_retries = 3
        self.retry_delay = 2
        self.timeout = 30
        self.chunk_size = 8192

    async def send_request_async(self) -> bool:
        """Enhanced asynchronous request with retries and better error handling"""
        logger.info(f'Sending request for user {self.user_id}...')
        self.stats.stage = "Fetching playlist data"
        self.stats.last_update = time.time()
        
        for attempt in range(self.max_retries):
            try:
                connector = aiohttp.TCPConnector(
                    limit=100,
                    limit_per_host=10,
                    keepalive_timeout=30,
                    enable_cleanup_closed=True
                )
                
                timeout = aiohttp.ClientTimeout(
                    total=self.timeout,
                    connect=10,
                    sock_read=10
                )
                
                async with aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': 'application/json,*/*',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Referer': 'https://vimeo.com/',
                        'Origin': 'https://vimeo.com'
                    }
                ) as session:
                    async with session.get(self.playlist_url, ssl=False) as resp:
                        if resp.status == 200:
                            content = await resp.text()
                            try:
                                self.response_data = json.loads(content)
                            except json.JSONDecodeError:
                                # Some Vimeo JSONs may be embedded - try to extract
                                try:
                                    match = re.search(r'({".*"})', content, re.DOTALL)
                                    if match:
                                        self.response_data = json.loads(match.group(1))
                                except Exception:
                                    raise
                            logger.info(f"Successfully fetched playlist for user {self.user_id}")
                            return True
                        else:
                            logger.warning(f"HTTP {resp.status} for user {self.user_id}, attempt {attempt + 1}")
                            
            except (aiohttp.ClientError, json.JSONDecodeError, asyncio.TimeoutError) as e:
                logger.error(f"Request attempt {attempt + 1} failed for user {self.user_id}: {e}")
                self.stats.error_count += 1
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                
        return False

    def parse_playlist(self) -> bool:
        """Enhanced playlist parsing with better error handling and validation"""
        logger.info(f'Parsing playlist JSON for user {self.user_id}...')
        self.stats.stage = "Parsing media streams"
        self.stats.last_update = time.time()
        
        try:
            parsed = self.response_data
            
            if not parsed or not isinstance(parsed, dict):
                logger.error(f"Invalid playlist data for user {self.user_id}")
                return False
            
            # try multiple keys (flexible parsing)
            self.clip_id = parsed.get('clip_id') or parsed.get('id') or parsed.get('video_id')
            if not self.clip_id:
                # try nested
                for key in ['video', 'clip', 'data']:
                    if key in parsed and isinstance(parsed[key], dict):
                        self.clip_id = parsed[key].get('id') or parsed[key].get('clip_id')
                        if self.clip_id:
                            break
            if not self.clip_id:
                logger.error("No clip_id found")
                return False
            
            self.main_base = parsed.get('base_url', parsed.get('base', './'))
            self.video_streams = parsed.get('video', parsed.get('videos', [])) or []
            self.audio_streams = parsed.get('audio', parsed.get('audios', [])) or []
            
            if not self.video_streams:
                logger.error("No video streams found")
                return False
            
            # Normalize stream entries to have expected keys
            def norm_streams(streams, kind='video'):
                normalized = []
                for s in streams:
                    if not isinstance(s, dict):
                        continue
                    normalized.append(s)
                return normalized
            
            self.video_streams = sorted(norm_streams(self.video_streams, 'video'), key=lambda x: x.get('height', 0), reverse=True)
            self.audio_streams = sorted(norm_streams(self.audio_streams, 'audio'), key=lambda x: x.get('bitrate', 0), reverse=True)
            
            logger.info(f"Found {len(self.video_streams)} video streams and {len(self.audio_streams)} audio streams for user {self.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Playlist parsing failed for user {self.user_id}: {e}")
            return False

    def get_available_qualities(self) -> List[str]:
        """Get list of available video qualities"""
        qualities = []
        for v in self.video_streams:
            height = v.get('height', 0)
            fps = v.get('fps', 0)
            quality = f"{height}p"
            if fps and fps > 30:
                quality += f"@{int(fps)}"
            qualities.append(quality)
        return qualities

    def _generate_variant_m3u8(self, stream: Dict, is_audio: bool = False) -> str:
        """Generate variant m3u8 content for a stream"""
        base_url = urljoin(self.main_base + "/", stream.get('base_url', ''))
        init_segment = stream.get('init_segment', '')
        segments = stream.get('segments', [])
        
        m3u8 = "#EXTM3U\n#EXT-X-VERSION:6\n"
        if init_segment:
            m3u8 += f'#EXT-X-MAP:URI="data:application/vnd.apple.mpegurl;base64,{init_segment}"\n'
        
        for seg in segments:
            duration = (seg.get('duration', 0) / 1000) if seg.get('duration') else 0.0
            url = urljoin(base_url, seg.get('url', ''))
            m3u8 += f"#EXTINF:{duration:.3f},\n{url}\n"
        
        m3u8 += "#EXT-X-ENDLIST\n"
        return m3u8

    async def create_m3u8_files(self, temp_dir: str, selected_video_index: int = 0, selected_audio_index: int = 0) -> str:
        """Create local m3u8 files for downloading"""
        video = self.video_streams[selected_video_index]
        audio = self.audio_streams[selected_audio_index] if self.audio_streams else None
        
        master_m3u8_path = os.path.join(temp_dir, 'master.m3u8')
        video_m3u8_path = os.path.join(temp_dir, 'video.m3u8')
        
        video_m3u8 = self._generate_variant_m3u8(video)
        async with aiofiles.open(video_m3u8_path, 'w') as f:
            await f.write(video_m3u8)
        
        master_m3u8 = "#EXTM3U\n#EXT-X-VERSION:6\n"
        bandwidth = int(video.get('avg_bitrate', video.get('bitrate', 0)))
        resolution = f"{video.get('width', 0)}x{video.get('height', 0)}"
        codecs = "avc1.4d401f,mp4a.40.2"
        
        if audio:
            audio_m3u8_path = os.path.join(temp_dir, 'audio.m3u8')
            audio_m3u8 = self._generate_variant_m3u8(audio, is_audio=True)
            async with aiofiles.open(audio_m3u8_path, 'w') as f:
                await f.write(audio_m3u8)
            
            master_m3u8 += '#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="audio",NAME="Main",DEFAULT=YES,AUTOSELECT=YES,URI="audio.m3u8"\n'
            master_m3u8 += f'#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},RESOLUTION={resolution},CODECS="{codecs}",AUDIO="audio"\nvideo.m3u8\n'
        else:
            master_m3u8 += f'#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},RESOLUTION={resolution},CODECS="{codecs}"\nvideo.m3u8\n'
        
        async with aiofiles.open(master_m3u8_path, 'w') as f:
            await f.write(master_m3u8)
        
        # estimate segments count
        self.stats.total_segments = len(video.get('segments', []))
        return master_m3u8_path

    async def download(self, quality: str = "highest") -> Tuple[bool, str]:
        """Download the video with selected quality"""
        try:
            self.stats.stage = "Preparing download"
            if quality == "highest":
                video_index = 0
            else:
                # allow qualities like "720p" or "720p@60"
                height_part = quality.split('p')[0]
                try:
                    height = int(height_part.split('@')[0])
                except Exception:
                    height = None
                if height:
                    video_index = next((i for i, v in enumerate(self.video_streams) if v.get('height') == height), 0)
                else:
                    video_index = 0
            
            audio_index = 0
            
            with tempfile.TemporaryDirectory(prefix=TEMP_DIR_PREFIX) as temp_dir:
                m3u8_path = await self.create_m3u8_files(temp_dir, video_index, audio_index)
                
                output_file = os.path.join(self.output_path, f"{self.clip_id}.mp4")
                cmd = [
                    N_M3U8DL_RE_PATH,
                    m3u8_path,
                    "--auto-select",
                    "--save-dir", self.output_path,
                    "--save-name", str(self.clip_id),
                    "--thread-count", "16",
                    "--mux-after-done", "format=mp4:muxer=ffmpeg"
                ]
                
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # read stdout to estimate progress
                self.stats.total_segments = len(self.video_streams[video_index].get('segments', []))
                self.stats.stage = "Downloading"
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    try:
                        decoded = line.decode().strip()
                    except Exception:
                        decoded = ""
                    # crude heuristics to update progress
                    if "Downloading" in decoded or "downloaded" in decoded.lower() or "segment" in decoded.lower():
                        self.stats.current_segment = min(self.stats.current_segment + 1, max(1, self.stats.total_segments))
                        progress = (self.stats.current_segment / max(1, self.stats.total_segments))
                        # estimate bytes if total unknown
                        estimated_total = self.stats.total_bytes or (50 * 1024 * 1024)
                        self.stats.bytes_downloaded = int(progress * estimated_total)
                        self.stats.last_update = time.time()
                
                await process.wait()
                
                if process.returncode == 0:
                    self.stats.stage = "Completed"
                    db_manager.log_download(
                        user_id=self.user_id,
                        url=self.playlist_url,
                        title=str(self.clip_id),
                        file_size=self.stats.bytes_downloaded,
                        download_speed=self.stats.speed_mbps,
                        quality=quality,
                        format_type="mp4",
                        status="completed"
                    )
                    return True, output_file
                else:
                    self.stats.stage = "Failed"
                    error_message = (await process.stderr.read()).decode()
                    db_manager.log_error(self.user_id, "DownloadError", error_message, url=self.playlist_url)
                    return False, error_message
            
        except Exception as e:
            logger.error(f"Download failed for user {self.user_id}: {e}")
            db_manager.log_error(self.user_id, "DownloadException", str(e), url=self.playlist_url)
            self.stats.stage = "Failed"
            return False, str(e)

async def progress_updater(user_id: int, chat_id: int, initial_message: Message = None):
    """Update download progress periodically by editing a single message.
       - Only edits message when progress changes by >= MIN_PROGRESS_DELTA or stage changes.
       - Stops when stage is Completed or Failed."""
    last_progress = -1.0
    last_stage = None
    last_edit_time = 0.0
    message_obj = initial_message  # Message object to edit
    # If we don't have initial message, we'll post one and then edit it
    if message_obj is None:
        # create a dummy minimal Message-like object by sending a message to the chat
        try:
            # the bot's Application is not globally available in this function, we'll use stored message id map
            # if no message is present, nothing to edit — bail out
            return
        except Exception:
            return

    while True:
        if user_id not in active_downloads:
            # no active download, finalize message and exit
            try:
                await send_markdown_message(message_obj, f"Download session ended or cleared.")
            except Exception:
                pass
            break

        stats = active_downloads.get(user_id)
        if not stats:
            break

        progress = (stats.current_segment / stats.total_segments) if stats.total_segments > 0 else 0.0
        # decide whether to edit:
        time_now = time.time()
        progress_delta = abs(progress - last_progress)
        stage_changed = (stats.stage != last_stage)

        should_edit = False
        if stage_changed:
            should_edit = True
        elif progress_delta >= MIN_PROGRESS_DELTA and (time_now - last_edit_time) >= PROGRESS_UPDATE_INTERVAL:
            should_edit = True
        elif (time_now - last_edit_time) >= (PROGRESS_UPDATE_INTERVAL * 5):
            # periodic heartbeat even if tiny changes (every 5 intervals)
            should_edit = True

        if should_edit:
            bytes_per_sec = stats.speed_mbps * 1024 * 1024 if stats.speed_mbps else 0
            message_text = (
                f"Download Progress: {escape_markdown(stats.stage)}\n"
                f"{create_progress_bar(progress)}\n"
                f"Downloaded: {format_size(stats.bytes_downloaded)} / {format_size(stats.total_bytes)}\n"
                f"Speed: {format_speed(bytes_per_sec)}\n"
                f"ETA: {format_time(stats.eta_seconds)}\n\n"
                f"{generate_random_tips()}"
            )
            try:
                await send_markdown_message(message_obj, message_text)
            except Exception as e:
                logger.debug(f"Failed to edit progress message for user {user_id}: {e}")
            last_progress = progress
            last_stage = stats.stage
            last_edit_time = time_now

        # stop if completed or failed
        if stats.stage in ("Completed", "Failed"):
            final_text = f"Download {stats.stage}.\nDownloaded: {format_size(stats.bytes_downloaded)}"
            try:
                await send_markdown_message(message_obj, final_text)
            except Exception:
                pass
            # clean active download state
            try:
                if user_id in active_downloads:
                    del active_downloads[user_id]
            except Exception:
                pass
            break

        await asyncio.sleep(PROGRESS_UPDATE_INTERVAL)

async def start(update: Update, context: CallbackContext) -> None:
    """Handle the /start command"""
    user = update.effective_user
    if not user:
        return
    # basic premium detection fallback
    is_premium = False
    try:
        is_premium = bool(user and getattr(user, "is_premium", False))
    except Exception:
        is_premium = False

    db_manager.update_user(
        user_id=user.id,
        username=user.username or "",
        first_name=user.first_name or "",
        last_name=user.last_name or "",
        is_premium=is_premium
    )
    # create user session record if not present
    if user.id not in user_sessions:
        user_sessions[user.id] = UserSession(user_id=user.id, username=user.username or "", is_premium=is_premium)

    await send_markdown_message(update, "Welcome to the Vimeo Downloader Bot! Send a Vimeo URL to begin.")

async def handle_message(update: Update, context: CallbackContext) -> None:
    """Handle incoming messages (e.g., Vimeo URLs)"""
    if not update.message:
        return
    url = update.message.text.strip()
    user_id = update.effective_user.id
    
    if user_id in active_downloads and active_downloads[user_id].stage != "Completed":
        await send_markdown_message(update, "Please wait for your current download to complete.")
        return
    
    user_stats = db_manager.get_user_stats(user_id)
    session = user_sessions.get(user_id) or UserSession(user_id=user_id, username=update.effective_user.username or "", is_premium=False)
    user_limit = PREMIUM_USER_LIMIT if session.is_premium else FREE_USER_LIMIT
    if user_stats.get('bytes_today', 0) >= user_limit:
        await send_markdown_message(update, "Daily download limit reached! Try again tomorrow or upgrade to premium.")
        return
    
    temp_dir = tempfile.mkdtemp(prefix=TEMP_DIR_PREFIX)
    downloader = EnhancedVimeoDownloader(url, temp_dir, user_id)
    # store for later selection handling
    session_downloaders[user_id] = downloader
    
    await update.message.reply_chat_action(ChatAction.TYPING)
    if await downloader.send_request_async() and downloader.parse_playlist():
        qualities = downloader.get_available_qualities()
        if qualities:
            keyboard = [[InlineKeyboardButton(q, callback_data=f"quality_{q}_{user_id}")] for q in qualities]
            reply_markup = InlineKeyboardMarkup(keyboard)
            msg = await send_markdown_message(update, f"Available qualities:\n{', '.join(qualities)}", reply_markup=reply_markup)
            
            # start or restart progress updater tied to this message (we will edit this message as the progress target)
            if user_id in progress_tasks:
                try:
                    progress_tasks[user_id].cancel()
                except Exception:
                    pass
                del progress_tasks[user_id]
            # store message info to be used by progress_updater (we pass msg)
            if msg:
                progress_tasks[user_id] = asyncio.create_task(progress_updater(user_id, msg.chat_id, initial_message=msg))
        else:
            await send_markdown_message(update, "No video streams found!")
    else:
        await send_markdown_message(update, "Failed to fetch or parse video data.")
        db_manager.log_error(user_id, "FetchError", "Failed to fetch or parse video data", url=url)

async def handle_quality_selection(update: Update, context: CallbackContext) -> None:
    """Handle quality selection"""
    query = update.callback_query
    if not query or not query.data:
        return
    # data format: "quality_{quality}_{user_id}"
    parts = query.data.split('_')
    if len(parts) < 3:
        await send_markdown_message(query, "Invalid selection data.")
        return
    _, quality_raw, user_id_part = parts[0], parts[1], parts[2]
    quality = quality_raw
    try:
        user_id = int(user_id_part)
    except Exception:
        await send_markdown_message(query, "Invalid user id in selection.")
        return
    
    if user_id not in session_downloaders:
        await send_markdown_message(query, "Session expired. Please send the URL again.")
        return
    
    downloader = session_downloaders.get(user_id)
    if not downloader:
        await send_markdown_message(query, "Session expired. Please send the URL again.")
        return

    # cancel existing progress updater if any, we'll start a new one pointing at this callback's message
    if user_id in progress_tasks:
        try:
            progress_tasks[user_id].cancel()
        except Exception:
            pass
        del progress_tasks[user_id]

    await query.message.reply_chat_action(ChatAction.UPLOAD_VIDEO)
    # start progress updater that will edit this callback message; create a copy of message to edit
    progress_tasks[user_id] = asyncio.create_task(progress_updater(user_id, query.message.chat_id, initial_message=query.message))

    success, result = await downloader.download(quality=quality)
    
    if success:
        await send_markdown_message(query, f"✅ Download completed for {quality}!")
        # try to send the resulting file (non-blocking)
        try:
            await query.message.reply_chat_action(ChatAction.UPLOAD_DOCUMENT)
            # reply with a document using the file path
            await query.message.reply_document(document=open(result, 'rb'), caption=f"Downloaded {quality} video")
        except Exception as e:
            logger.error(f"Failed to send video for user {user_id}: {e}")
            await send_markdown_message(query, "Download completed but failed to send video. You can retrieve it from the server.")
    else:
        await send_markdown_message(query, f"❌ Download failed: {escape_markdown(str(result))}")
    
    # cleanup tasks and state
    if user_id in progress_tasks:
        try:
            progress_tasks[user_id].cancel()
        except Exception:
            pass
        del progress_tasks[user_id]
    if user_id in active_downloads:
        try:
            del active_downloads[user_id]
        except Exception:
            pass
    if user_id in session_downloaders:
        try:
            del session_downloaders[user_id]
        except Exception:
            pass

async def speedtest_command(update: Update, context: CallbackContext) -> None:
    """Run speedtest and report results"""
    if not update.message:
        return
    await update.message.reply_chat_action(ChatAction.TYPING)
    try:
        st = speedtest.Speedtest()
        st.get_best_server()
        download_speed = st.download() / 1024 / 1024  # Convert to Mbps
        upload_speed = st.upload() / 1024 / 1024  # Convert to Mbps
        ping = st.results.ping
        
        message = (
            f"Speedtest Results:\n"
            f"Download: {download_speed:.2f} Mbps\n"
            f"Upload: {upload_speed:.2f} Mbps\n"
            f"Ping: {ping:.2f} ms"
        )
        await send_markdown_message(update, message)
    except Exception as e:
        logger.error(f"Speedtest failed: {e}")
        await send_markdown_message(update, "Speedtest failed. Please try again later.")

async def stats_command(update: Update, context: CallbackContext) -> None:
    """Show user statistics"""
    user_id = update.effective_user.id if update.effective_user else None
    if not user_id:
        return
    stats = db_manager.get_user_stats(user_id)
    
    message = (
        f"Your Statistics:\n"
        f"Total Downloads: {stats.get('total_downloads', 0)}\n"
        f"Successful Downloads: {stats.get('successful_downloads', 0)}\n"
        f"Failed Downloads: {stats.get('failed_downloads', 0)}\n"
        f"Total Data: {format_size(stats.get('total_bytes', 0))}\n"
        f"Today's Data: {format_size(stats.get('bytes_today', 0))}\n"
        f"Average Speed: {format_speed(stats.get('avg_speed', 0) * 1024 * 1024)}\n"
        f"Success Rate: {stats.get('success_rate', 0):.1f}%"
    )
    await send_markdown_message(update, message)

async def main():
    """Start the bot"""
    application = Application.builder().token(BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("speedtest", speedtest_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(handle_quality_selection, pattern="^quality_"))
    
    # run polling (non-blocking)
    await application.run_polling()

if __name__ == "__main__":
    nest_asyncio.apply()
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Bot stopped by user")
