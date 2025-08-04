import json
import os
import logging
from fastapi import FastAPI, Request, HTTPException
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode
from datetime import datetime, timedelta
import uuid
import mimetypes
from threading import Lock
from aiofiles import os as aio_os
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio
from dotenv import load_dotenv
import uvicorn
from uvicorn.config import Config
from uvicorn.server import Server

# Load .env file for local development test
if os.getenv("WEBHOOK_URL", "").startswith("http://localhost"):
    load_dotenv()

# Configure logging
def configure_logging():
    enable_logging = os.getenv("ENABLE_LOGGING", "true").lower() == "true"
    log_level = logging.INFO if enable_logging else logging.WARNING
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    logger = logging.getLogger(__name__)
    return logger

logger = configure_logging()

# Log environment variables
def log_env_variables():
    bot_token = os.getenv("BOT_TOKEN")
    webhook_url = os.getenv("WEBHOOK_URL")
    port = os.getenv("PORT", "8000")
    masked_token = f"{bot_token[:5]}...{bot_token[-5:]}" if bot_token else "Not set"
    logger.info(f"Environment Variables: BOT_TOKEN={masked_token}, WEBHOOK_URL={webhook_url}, PORT={port}")

# File and directory constants
DATA_FILE = "group_data.json"
MEDIA_DIR = "media"

# Ensure media directory exists
os.makedirs(MEDIA_DIR, exist_ok=True)

# Thread lock for job queue operations
job_lock = Lock()

# Load group data from JSON
def load_group_data():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding group_data.json: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading group_data.json: {e}")
            return {}
    return {}

# Save group data to JSON
def save_group_data(data):
    try:
        with open(DATA_FILE, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving group_data.json: {e}")

# Ensure group is initialized
def ensure_group_initialized(chat_id, group_data):
    chat_id = str(chat_id)
    if chat_id not in group_data:
        group_data[chat_id] = {"messages": [], "pending_message": None}
        save_group_data(group_data)
    return group_data

# Check if bot is admin
async def check_admin_permissions(chat_id, bot: Bot):
    try:
        member = await bot.get_chat_member(chat_id, bot.id)
        return member.status in ("administrator", "creator")
    except Exception as e:
        logger.error(f"Error checking admin status for chat {chat_id}: {e}")
        return False

# Validate message text length
def validate_message_text(text):
    if len(text) > 4096:  # Telegram's message limit
        return False, "Message text too long (max 4096 characters)"
    if not text.strip():
        return False, "Message text cannot be empty"
    return True, None

# Escape MarkdownV2 special characters
def escape_markdown_v2(text):
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text

# FastAPI app
app = FastAPI()

# Initialize bot and application
bot_token = os.getenv("BOT_TOKEN")
if not bot_token:
    logger.error("BOT_TOKEN environment variable not set or empty.")
    raise ValueError("BOT_TOKEN environment variable not set or empty.")

# Validate token format
if not bot_token.strip() or ' ' in bot_token:
    logger.error("BOT_TOKEN is invalid: contains spaces or is empty.")
    raise ValueError("BOT_TOKEN is invalid: contains spaces or is empty.")

if not bot_token.count(':') == 1 or not bot_token.split(':')[0].isdigit():
    logger.error("BOT_TOKEN is invalid: must follow format <number>:<alphanumeric>.")
    raise ValueError("BOT_TOKEN is invalid: must follow format <number>:<alphanumeric>.")

application = Application.builder().token(bot_token).build()

# Help command: List all commands with descriptions
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    if not await check_admin_permissions(chat_id, context.bot):
        await update.message.reply_text(
            escape_markdown_v2("⚠️ Please make me an admin to use commands!"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    response = (
        "🤖 **@QSWAPCommunityBot Commands**:\n\n"
        "📝 /start - Initialize the bot and show available commands\n"
        "📩 /addmessage <interval_minutes> <text> - Add a new message with optional media\n"
        "❌ /cancel - Cancel pending message and add as text-only\n"
        "📋 /listmessages - List all configured messages\n"
        "🗑️ /deletemessage <number> - Delete a specific message\n"
        "🔍 /printjson - Show group data in JSON format\n"
        "🆔 /getchatid - Get the current chat ID\n"
        "🧪 /testpost <number> - Test a specific message\n"
        "📅 /nextpost - Show next scheduled time for each message"
    )
    await update.message.reply_text(
        escape_markdown_v2(response),
        parse_mode=ParseMode.MARKDOWN_V2
    )

# Start command: Initialize group
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    group_data = ensure_group_initialized(chat_id, load_group_data())

    if not await check_admin_permissions(chat_id, context.bot):
        await update.message.reply_text(
            escape_markdown_v2("⚠️ Please make me an admin to post messages!"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    await update.message.reply_text(
        escape_markdown_v2(
            "🤖 Bot started!\n\n"
            "Use /help to see all available commands."
        ),
        parse_mode=ParseMode.MARKDOWN_V2
    )

# Add a message (prompt for media)
async def add_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    group_data = ensure_group_initialized(chat_id, load_group_data())

    args = update.message.text.split(maxsplit=2)[1:] if len(update.message.text.split()) > 1 else []
    if len(args) < 2:
        await update.message.reply_text(
            escape_markdown_v2("❌ Usage: /addmessage <interval_minutes> <text>"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    if not args[0].isdigit():
        await update.message.reply_text(
            escape_markdown_v2("❌ Interval must be a number (minutes)"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    interval = int(args[0])
    if interval < 1:
        await update.message.reply_text(
            escape_markdown_v2("❌ Interval must be at least 1 minute"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    if interval > 10080:  # 1 week in minutes
        await update.message.reply_text(
            escape_markdown_v2("❌ Interval cannot exceed 1 week (10080 minutes)"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    if len(group_data[chat_id]["messages"]) >= 10:
        await update.message.reply_text(
            escape_markdown_v2("❌ Maximum 10 messages allowed. Delete one with /listmessages first."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    text = args[1]
    is_valid, error_msg = validate_message_text(text)
    if not is_valid:
        await update.message.reply_text(
            escape_markdown_v2(f"❌ {error_msg}"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    group_data[chat_id]["pending_message"] = {"text": text, "interval_minutes": interval}
    save_group_data(group_data)
    await update.message.reply_text(
        escape_markdown_v2(
            f"✅ Message prepared!\n\n"
            f"📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}\n"
            f"⏰ Interval: {interval} minutes\n\n"
            f"📷 Send a photo (JPEG/PNG) or video (MP4/GIF), or use /cancel for text-only message."
        ),
        parse_mode=ParseMode.MARKDOWN_V2
    )

# Cancel pending message (add as text-only)
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    group_data = ensure_group_initialized(chat_id, load_group_data())

    if not group_data[chat_id]["pending_message"]:
        await update.message.reply_text(
            escape_markdown_v2("❌ No pending message to cancel."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    pending_message = group_data[chat_id]["pending_message"]
    text = pending_message["text"]
    interval = pending_message["interval_minutes"]

    group_data[chat_id]["messages"].append({
        "text": text,
        "media": None,
        "interval_minutes": interval
    })
    group_data[chat_id]["pending_message"] = None
    save_group_data(group_data)

    await start_posting_job(chat_id, context.bot, interval, len(group_data[chat_id]["messages"]) - 1)
    await update.message.reply_text(
        escape_markdown_v2(f"✅ Text-only message added with {interval}-minute interval!"),
        parse_mode=ParseMode.MARKDOWN_V2
    )

# Handle media uploads
async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    group_data = ensure_group_initialized(chat_id, load_group_data())

    if not group_data[chat_id]["pending_message"]:
        await update.message.reply_text(
            escape_markdown_v2("❌ No pending message. Use /addmessage first."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    pending_message = group_data[chat_id]["pending_message"]
    text = pending_message["text"]
    interval = pending_message["interval_minutes"]

    file = None
    file_ext = None
    file_size = 0

    if update.message.photo:
        file = await update.message.photo[-1].get_file()
        file_ext = ".jpg"  # Telegram photos are typically JPEG
        file_size = update.message.photo[-1].file_size
    elif update.message.video:
        file = await update.message.video.get_file()
        mime_type = update.message.video.mime_type
        file_ext = mimetypes.guess_extension(mime_type) or ".mp4"
        file_size = update.message.video.file_size

    if not file:
        await update.message.reply_text(
            escape_markdown_v2("❌ Please send a photo (JPEG/PNG) or video (MP4/GIF)."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    # Check file size (Telegram limit is 20MB for bots)
    if file_size and file_size > 20 * 1024 * 1024:
        await update.message.reply_text(
            escape_markdown_v2("❌ File too large. Maximum size is 20MB."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    file_name = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(MEDIA_DIR, file_name)

    try:
        await update.message.reply_text(
            escape_markdown_v2("⏳ Downloading media..."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        await file.download_to_drive(file_path)
    except Exception as e:
        logger.error(f"Error downloading file for chat {chat_id}: {e}")
        await update.message.reply_text(
            escape_markdown_v2("❌ Error saving media. Try again or use /cancel."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    group_data[chat_id]["messages"].append({
        "text": text,
        "media": file_path,
        "interval_minutes": interval
    })
    group_data[chat_id]["pending_message"] = None
    save_group_data(group_data)

    await start_posting_job(chat_id, context.bot, interval, len(group_data[chat_id]["messages"]) - 1)
    await update.message.reply_text(
        escape_markdown_v2(f"✅ Message with media added! Interval: {interval} minutes"),
        parse_mode=ParseMode.MARKDOWN_V2
    )

# List messages
async def list_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    group_data = ensure_group_initialized(chat_id, load_group_data())

    if not group_data[chat_id]["messages"]:
        await update.message.reply_text(
            escape_markdown_v2("📝 No messages configured yet. Use /addmessage to add one!"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    response = "📋 **Current Messages:**\n\n"
    for i, msg in enumerate(group_data[chat_id]["messages"]):
        media_info = "📷 Photo" if msg['media'] and msg['media'].endswith(('.jpg', '.jpeg', '.png')) else \
            "🎥 Video" if msg['media'] and msg['media'].endswith(('.mp4', '.gif')) else \
                "📝 Text only"

        response += (f"**{i + 1}.** {escape_markdown_v2(msg['text'][:50])}{'...' if len(msg['text']) > 50 else ''}\n"
                     f"   📁 Media: {escape_markdown_v2(media_info)}\n"
                     f"   ⏰ Interval: {msg['interval_minutes']} minutes\n\n")

    response += "🗑️ To delete: /deletemessage <number>"

    # Split long messages
    if len(response) > 4096:
        parts = [response[i:i + 4000] for i in range(0, len(response), 4000)]
        for part in parts:
            await update.message.reply_text(
                escape_markdown_v2(part),
                parse_mode=ParseMode.MARKDOWN_V2
            )
    else:
        await update.message.reply_text(
            escape_markdown_v2(response),
            parse_mode=ParseMode.MARKDOWN_V2
        )

# Delete a message
async def delete_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    group_data = ensure_group_initialized(chat_id, load_group_data())

    args = update.message.text.split(maxsplit=1)[1:] if len(update.message.text.split()) > 1 else []
    if not args or not args[0].isdigit():
        await update.message.reply_text(
            escape_markdown_v2("❌ Usage: /deletemessage <number>"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    index = int(args[0]) - 1
    if 0 <= index < len(group_data[chat_id]["messages"]):
        deleted_message = group_data[chat_id]["messages"].pop(index)

        # Clean up media file
        if deleted_message["media"] and os.path.exists(deleted_message["media"]):
            try:
                await aio_os.remove(deleted_message["media"])
            except Exception as e:
                logger.error(f"Error deleting media file {deleted_message['media']}: {e}")

        save_group_data(group_data)

        # Remove associated job
        scheduler = context.bot_data.get('scheduler')
        job_id = f"{chat_id}_{index}"
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)
            logger.info(f"Removed job {job_id} for chat {chat_id}")

        await update.message.reply_text(
            escape_markdown_v2(f"✅ Message {index + 1} deleted successfully!"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
    else:
        await update.message.reply_text(
            escape_markdown_v2("❌ Invalid message number. Use /listmessages to see available messages."),
            parse_mode=ParseMode.MARKDOWN_V2
        )

# Print JSON data
async def print_json(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    group_data = ensure_group_initialized(chat_id, load_group_data())

    clean_data = {"messages": []}
    for msg in group_data[chat_id]["messages"]:
        clean_msg = {
            "text": msg["text"],
            "has_media": msg["media"] is not None,
            "interval_minutes": msg["interval_minutes"]
        }
        clean_data["messages"].append(clean_msg)

    formatted_data = json.dumps(clean_data, indent=2, ensure_ascii=False)
    await update.message.reply_text(
        escape_markdown_v2(f"📊 **Group Data:**\n```json\n{formatted_data}\n```"),
        parse_mode=ParseMode.MARKDOWN_V2
    )

# Get chat ID
async def get_chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    chat_type = update.effective_chat.type
    chat_title = getattr(update.effective_chat, 'title', 'N/A')

    await update.message.reply_text(
        escape_markdown_v2(
            f"🆔 **Chat Information:**\n"
            f"ID: `{chat_id}`\n"
            f"Type: {chat_type}\n"
            f"Title: {chat_title}"
        ),
        parse_mode=ParseMode.MARKDOWN_V2
    )

# Show next scheduled post times
async def next_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    group_data = ensure_group_initialized(chat_id, load_group_data())

    if not group_data[chat_id]["messages"]:
        await update.message.reply_text(
            escape_markdown_v2("📝 No messages configured yet. Use /addmessage to add one!"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    scheduler = context.bot_data.get('scheduler')
    response = "📅 **Next Scheduled Posts:**\n\n"
    for i, msg in enumerate(group_data[chat_id]["messages"]):
        job_id = f"{chat_id}_{i}"
        job = scheduler.get_job(job_id)
        if job:
            next_run = job.next_run_time
            next_run_str = next_run.strftime("%Y-%m-%d %H:%M:%S %Z")
            response += (
                f"**Message {i + 1}:** {escape_markdown_v2(msg['text'][:50])}{'...' if len(msg['text']) > 50 else ''}\n"
                f"   ⏰ Next run: {escape_markdown_v2(next_run_str)}\n"
                f"   🔄 Interval: {msg['interval_minutes']} minutes\n\n"
            )
        else:
            response += (
                f"**Message {i + 1}:** {escape_markdown_v2(msg['text'][:50])}{'...' if len(msg['text']) > 50 else ''}\n"
                f"   ⚠️ No active job (restarting job...)\n"
                f"   🔄 Interval: {msg['interval_minutes']} minutes\n\n"
            )
            # Restart job if missing
            await start_posting_job(chat_id, context.bot, msg["interval_minutes"], i)

    # Split long messages
    if len(response) > 4096:
        parts = [response[i:i + 4000] for i in range(0, len(response), 4000)]
        for part in parts:
            await update.message.reply_text(
                escape_markdown_v2(part),
                parse_mode=ParseMode.MARKDOWN_V2
            )
    else:
        await update.message.reply_text(
            escape_markdown_v2(response),
            parse_mode=ParseMode.MARKDOWN_V2
        )

# Test post a specific message
async def test_post(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)
    group_data = ensure_group_initialized(chat_id, load_group_data())

    args = update.message.text.split(maxsplit=1)[1:] if len(update.message.text.split()) > 1 else []
    if not args or not args[0].isdigit():
        await update.message.reply_text(
            escape_markdown_v2("❌ Usage: /testpost <number>"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    index = int(args[0]) - 1
    if 0 <= index < len(group_data[chat_id]["messages"]):
        await update.message.reply_text(
            escape_markdown_v2("🧪 Testing message..."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        await post_message(context.bot, chat_id, index)
        await update.message.reply_text(
            escape_markdown_v2(f"✅ Test completed for message {index + 1}"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
    else:
        await update.message.reply_text(
            escape_markdown_v2("❌ Invalid message number. Use /listmessages to see available messages."),
            parse_mode=ParseMode.MARKDOWN_V2
        )

# Post message job
async def post_message(bot: Bot, chat_id: str, message_index: int):
    group_data = load_group_data()
    if chat_id not in group_data:
        logger.error(f"Chat {chat_id} not found in group_data")
        return
    if message_index >= len(group_data[chat_id]["messages"]):
        logger.error(f"Message index {message_index} out of range for chat {chat_id}")
        return

    message = group_data[chat_id]["messages"][message_index]
    try:
        if not await check_admin_permissions(chat_id, bot):
            logger.error(f"Bot is not admin in chat {chat_id}. Cannot post message.")
            return

        if message["media"]:
            if not os.path.exists(message["media"]):
                logger.error(f"Media file {message['media']} does not exist")
                return

            media_path = message["media"]
            if media_path.lower().endswith((".jpg", ".jpeg", ".png")):
                with open(media_path, "rb") as f:
                    await bot.send_photo(
                        chat_id=chat_id,
                        photo=f,
                        caption=escape_markdown_v2(message["text"]) if message["text"] else None,
                        parse_mode=ParseMode.MARKDOWN_V2
                    )
            elif media_path.lower().endswith((".mp4", ".gif")):
                with open(media_path, "rb") as f:
                    await bot.send_video(
                        chat_id=chat_id,
                        video=f,
                        caption=escape_markdown_v2(message["text"]) if message["text"] else None,
                        parse_mode=ParseMode.MARKDOWN_V2
                    )
            else:
                logger.warning(f"Unknown media type for file {media_path}, sending as document")
                with open(media_path, "rb") as f:
                    await bot.send_document(
                        chat_id=chat_id,
                        document=f,
                        caption=escape_markdown_v2(message["text"]) if message["text"] else None,
                        parse_mode=ParseMode.MARKDOWN_V2
                    )
        else:
            await bot.send_message(
                chat_id=chat_id,
                text=escape_markdown_v2(message["text"]),
                parse_mode=ParseMode.MARKDOWN_V2
            )
    except Exception as e:
        logger.error(f"Failed to post message to chat {chat_id}, index {message_index}: {str(e)}")

# Start posting job for a message
async def start_posting_job(chat_id: str, bot: Bot, interval: int, message_index: int):
    group_data = load_group_data()
    if chat_id not in group_data or message_index >= len(group_data[chat_id]["messages"]):
        logger.error(f"No message at index {message_index} for chat {chat_id}. Job not started.")
        return

    if not await check_admin_permissions(chat_id, bot):
        logger.error(f"Bot is not admin in chat {chat_id}. Skipping job scheduling.")
        return

    with job_lock:
        scheduler = application.bot_data.get('scheduler')
        job_id = f"{chat_id}_{message_index}"
        if scheduler.get_job(job_id):
            scheduler.remove_job(job_id)
            logger.info(f"Removed existing job {job_id} for chat {chat_id}")
        scheduler.add_job(
            post_message,
            'interval',
            minutes=interval,
            id=job_id,
            args=(bot, chat_id, message_index),
            next_run_time=datetime.now() + timedelta(minutes=1)  # Start after 1 minute
        )
        logger.info(
            f"Scheduled job {job_id} for chat {chat_id}, message {message_index + 1}, interval {interval} minutes")

# Initialize bot and setup webhook
async def on_startup():
    # Initialize application
    try:
        await application.initialize()
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}")
        raise

    # Initialize scheduler
    scheduler = AsyncIOScheduler()
    application.bot_data['scheduler'] = scheduler
    scheduler.start()
    logger.info("Scheduler started")

    # Reload jobs from group_data.json
    group_data = load_group_data()
    for chat_id, data in group_data.items():
        for i, message in enumerate(data["messages"]):
            await start_posting_job(chat_id, application.bot, message["interval_minutes"], i)

    # Set webhook
    webhook_url = os.getenv("WEBHOOK_URL")
    if not webhook_url:
        logger.error("WEBHOOK_URL environment variable not set.")
        raise ValueError("WEBHOOK_URL environment variable not set.")

    try:
        await application.bot.set_webhook(webhook_url)
        logger.info(f"Webhook set to {webhook_url}")
    except Exception as e:
        logger.error(f"Failed to set webhook: {str(e)}")
        raise

async def on_shutdown():
    # Stop scheduler
    scheduler = application.bot_data.get('scheduler')
    scheduler.shutdown()
    logger.info("Scheduler shut down")

    # Delete webhook and shutdown application
    await application.bot.delete_webhook()
    await application.shutdown()
    logger.info("Webhook deleted and application shut down")

# FastAPI webhook endpoint
@app.post("/webhook")
async def webhook(request: Request):
    try:
        if not application.updater:  # Check if application is initialized
            logger.warning("Application not initialized yet, returning 503")
            raise HTTPException(status_code=503, detail="Application is initializing, please try again later")
        
        update = await request.json()
        update = Update.de_json(update, application.bot)
        await application.process_update(update)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error processing webhook update: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Main application
async def main():
    # Log environment variables
    log_env_variables()

    # Register handlers
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("addmessage", add_message))
    application.add_handler(CommandHandler("cancel", cancel))
    application.add_handler(MessageHandler(filters.PHOTO | filters.VIDEO, handle_media))
    application.add_handler(CommandHandler("listmessages", list_messages))
    application.add_handler(CommandHandler("deletemessage", delete_message))
    application.add_handler(CommandHandler("printjson", print_json))
    application.add_handler(CommandHandler("getchatid", get_chat_id))
    application.add_handler(CommandHandler("nextpost", next_post))
    application.add_handler(CommandHandler("testpost", test_post))

    # Run startup with error handling
    try:
        await on_startup()
        logger.info("Startup completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

    # Start FastAPI server
    host = "0.0.0.0"
    port = int(os.getenv("PORT", 8000))
    config = Config(app=app, host=host, port=port)
    server = Server(config)
    await server.serve()

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except (KeyboardInterrupt, SystemExit):
        loop.run_until_complete(on_shutdown())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
