import os
import logging
import asyncio
from datetime import datetime
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.errors.rpcerrorlist import FloodWaitError, MessageTooLongError
from telethon.sessions import StringSession

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Generative AI not installed. Summary features disabled.")

# Load environment variables
load_dotenv()

# Logging
logger = logging.getLogger("TelegramManager")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class TelegramManager:
    def __init__(self):
        # Validate required environment variables
        self.api_id = self._get_env_int("TELEGRAM_API_ID")
        self.api_hash = self._get_env_str("TELEGRAM_API_HASH")
        
        # Fix session path to be in the same directory as the script
        self.session_name = self._get_session_path()
        self.target_id = os.getenv("TELEGRAM_TARGET_ID", "me")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_id or not self.api_hash:
            raise ValueError("TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in environment variables")

        # Initialize client but don't connect here
        self.client = None
        self._connected = False
        self._connection_lock = asyncio.Lock()  # Prevent race conditions
        
        # Initialize Gemini with better error handling
        self.llm = None
        if GEMINI_AVAILABLE and self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.llm = genai.GenerativeModel("gemini-1.5-flash")
                logger.info("Gemini AI initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini AI: {e}")
                self.llm = None
        else:
            if not GEMINI_AVAILABLE:
                logger.warning("Gemini AI library not available.")
            elif not self.gemini_api_key:
                logger.warning("GEMINI_API_KEY not provided. Summary features disabled.")

    def _get_session_path(self) -> str:
        """Get the proper session file path, checking multiple locations."""
        session_name = os.getenv("TELEGRAM_SESSION_NAME", "telegram_bot_session")
        
        # Check multiple possible locations for existing session file
        possible_paths = [
            Path(__file__).parent / f"{session_name}.session",  # src directory
            Path.cwd() / f"{session_name}.session",              # current working directory
            Path.cwd() / "src" / f"{session_name}.session",      # src from root
            Path(f"{session_name}.session")                      # relative to current location
        ]
        
        # Look for existing session file first
        for path in possible_paths:
            if path.exists():
                logger.info(f"Found existing session file at: {path}")
                return str(path)
        
        # If no existing session found, create in src directory
        default_path = Path(__file__).parent / f"{session_name}.session"
        logger.info(f"No existing session found. New session will be created at: {default_path}")
        return str(default_path)

    def _get_env_int(self, key: str) -> Optional[int]:
        """Safely get integer environment variable."""
        value = os.getenv(key)
        if not value:
            logger.error(f"Environment variable {key} is not set")
            return None
        try:
            return int(value)
        except ValueError:
            logger.error(f"Environment variable {key} must be a valid integer")
            return None

    def _get_env_str(self, key: str) -> Optional[str]:
        """Safely get string environment variable."""
        value = os.getenv(key)
        if not value:
            logger.error(f"Environment variable {key} is not set")
        return value

    async def _ensure_connected(self):
        """Ensure the client is connected, with proper loop handling (max_attempt = 3) with 2 sec delay."""
        async with self._connection_lock:
            if self._connected and self.client and self.client.is_connected():
                return

            max_retries = 3  # Number of connection attempts
            retry_delay = 2  # Initial delay in seconds

            for attempt in range(1, max_retries + 1):
                try:
                    # Create new client if needed
                    if not self.client:
                        if os.getenv("TELEGRAM_BOT_TOKEN"):
                            # Use bot authentication via token
                            self.client = TelegramClient(StringSession(), self.api_id, self.api_hash)
                            await self.client.start(bot_token=os.getenv("TELEGRAM_BOT_TOKEN"))
                            self._connected = True
                            logger.info("Telegram bot client connected successfully.")
                            return
                        else:
                            # Default to user login
                            self.client = TelegramClient(
                                self.session_name,
                                self.api_id,
                                self.api_hash,
                                sequential_updates=True,
                                timeout=self._get_env_int("TELEGRAM_TIMEOUT") or 30,
                                request_retries=3,
                                connection_retries=3,
                            )
                    # Connect if not already connected
                    if not self.client.is_connected():
                        await self.client.connect()
                        
                        if not await self.client.is_user_authorized() and not os.getenv("TELEGRAM_BOT_TOKEN"):
                            logger.error("User not authorized. Session file may be corrupted or invalid.")
                            logger.info("Please delete the session file and re-authenticate manually.")
                            raise Exception("Authentication required - please run authentication setup first")
                        
                        self._connected = True
                        logger.info("Telegram client connected and authorized successfully.")
                        return  # Connection successful, exit loop

                except TimeoutError as te:
                    logger.error(f"Telegram connection timeout (attempt {attempt}/{max_retries}): {te}")
                    self._connected = False
                    if self.client:
                        try:
                            await self.client.disconnect()
                        except:
                            pass
                        self.client = None
                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise  # Re-raise the exception after max retries

                except Exception as e:
                    logger.error(f"Failed to connect Telegram client (attempt {attempt}/{max_retries}): {type(e).__name__} - {e}")
                    self._connected = False
                    # Clean up failed client
                    if self.client:
                        try:
                            await self.client.disconnect()
                        except:
                            pass
                        self.client = None
                    if attempt < max_retries:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise  # Re-raise the exception after max retries

            # If the loop completes without a successful connection, raise an exception
            raise Exception("Failed to connect to Telegram after multiple retries.")

    async def close(self):
        """Stop the Telegram client with proper cleanup."""
        try:
            if self.client and self.client.is_connected():
                # Give time for pending operations if in process
                await asyncio.sleep(0.1)
                await self.client.disconnect()
                self._connected = False
                logger.info("Telegram client disconnected.")
                
                # Wait  for cleanup
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.warning(f"Error disconnecting Telegram client: {e}")
        
        # Cancel any remaining tasks if needed
        try:
            # Get current event loop
            loop = asyncio.get_running_loop()
            
            # Cancel pending tasks related to this client
            pending_tasks = [task for task in asyncio.all_tasks(loop) 
                           if not task.done() and 'MTProtoSender' in str(task)]
            
            if pending_tasks:
                logger.info(f"Cancelling {len(pending_tasks)} pending Telegram tasks...")
                for task in pending_tasks:
                    task.cancel()
                
                # Wait for cancellation to complete
                await asyncio.gather(*pending_tasks, return_exceptions=True)
                
        except Exception as e:
            logger.debug(f"Task cleanup warning: {e}")

    async def generate_summary(self, text: str) -> str:
        """Generate summary with comprehensive error handling."""
        if not self.llm:
            logger.warning("Gemini LLM is not configured. Returning original text.")
            return text

        if not text or not text.strip():
            logger.warning("Empty text provided for summary.")
            return "No content to summarize."

        # Limit input text length to avoid API limits(Gemini llm)
        max_input_length = 50000  
        if len(text) > max_input_length:
            text = text[:max_input_length] + "... [truncated]"
            logger.info(f"Text truncated to {max_input_length} characters for summary.")

        try:
            logger.info("Generating LLM summary...")
            
            # Prompt for the summarized summary of the report
            prompt = f"""Please provide a concise summary of the following text for a Telegram message. 
                        Keep it under 200 words and focus on the key points:{text}"""
            response = self.llm.generate_content(prompt)
            
            if response and hasattr(response, 'text') and response.text:
                summary = response.text.strip()
                logger.info("Summary generated successfully.")
                return summary
            else:
                logger.error("Empty response from Gemini API.")
                return "Failed to generate summary - empty response."
                
        except Exception as e:
            logger.error(f"Gemini API error: {type(e).__name__}: {e}")
            
            # Return a fallback summary based on error type
            if "quota" in str(e).lower() or "limit" in str(e).lower():
                return "API quota exceeded. Cannot generate summary."
            elif "authentication" in str(e).lower() or "api_key" in str(e).lower():
                return "Authentication error. Check API key."
            else:
                return f"Summary generation failed: {type(e).__name__}"

    async def send(self, message: str, use_summary=False, prefix="", max_retries=3):
        """Send message with improved error handling and retry logic."""
        if not message:
            logger.warning("Empty message provided. Nothing to send.")
            return False

        # Ensure we're connected
        await self._ensure_connected()

        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Generate summary if requested
                if use_summary:
                    original_length = len(message)
                    message = await self.generate_summary(message)
                    logger.info(f"Message summarized: {original_length} -> {len(message)} characters")

                # Apply prefix
                final_message = f"{prefix}{message}"
                
                # Check message length (Telegram limit is ~4096 characters)
                if len(final_message) > 4096:
                    logger.warning("Message exceeds Telegram limit. Truncating...")
                    final_message = final_message[:4090] + "... [truncated]"

                # Send the message
                await self.client.send_message(self.target_id, final_message)
                logger.info("Message sent to Telegram successfully.")
                return True

            except FloodWaitError as fw:
                logger.warning(f"Rate limited. Waiting for {fw.seconds} seconds...")
                await asyncio.sleep(fw.seconds)
                continue  # Don't increment retry_count for flood wait
                
            except MessageTooLongError:
                logger.error("⚠️ Message too long even after truncation.")
                # Try to send a shorter version
                if not use_summary:
                    logger.info("Attempting to send with summary enabled...")
                    return await self.send(message, use_summary=True, prefix=prefix, max_retries=max_retries-retry_count)
                else:
                    await self.client.send_message(self.target_id, f"{prefix} Message too long to send.")
                    return False
                    
            except Exception as e:
                retry_count += 1
                logger.error(f"Failed to send Telegram message (attempt {retry_count}/{max_retries}): {e}")
                
                if retry_count >= max_retries:
                    logger.error("Max retries reached. Message send failed.")
                    return False
                    
                # Wait before retry
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
        
        return False

    async def send_error_alert(self, error_message: str, context: str = ""):
        """Send an error alert with standardized formatting."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_message = f" ERROR ALERT \n"
        alert_message += f"Time: {timestamp}\n"
        if context:
            alert_message += f"Context: {context}\n"
        alert_message += f"Error: {error_message}"
        
        await self.send(alert_message, use_summary=True, prefix="")

    async def health_check(self):
        """Perform a health check of the bot."""
        try:
            await self._ensure_connected()
            me = await self.client.get_me()

            if me is None:
                logger.error("Health check failed: get_me() returned None")
                return False

            if me.bot:
                logger.info(f"Health check passed. Connected as bot: @{me.username}")

            else:
                logger.info(f"Health check passed. Connected as: {me.first_name} (@{me.username or 'no_username'})")

            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    # Context manager support for cleaner usage
    async def __aenter__(self):
        await self._ensure_connected()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        # Additional cleanup delay
        await asyncio.sleep(0.2)


# Utility function for one-off message sending
async def send_telegram_message(message: str, use_summary=False, prefix=""):
    """Utility function to send a single message without managing the client lifecycle."""
    tg = None
    try:
        tg = TelegramManager()
        await tg._ensure_connected()
        result = await tg.send(message, use_summary=use_summary, prefix=prefix)
        return result
    finally:
        if tg:
            await tg.close()
            # Extra cleanup time
            await asyncio.sleep(0.3)


# --- Example usage and test ---
if __name__ == "__main__":
    async def test_bot():
        """Test the bot with various scenarios."""
        async with TelegramManager() as tg:
            try:
                # Health check
                if not await tg.health_check():
                    logger.error("Health check failed. Exiting.")
                    return
                
                # Test basic message
                await tg.send("Bot startup test successful!")
                
                # Test summary with long message
                long_message = "This is a simulated error traceback:\n" + "\n".join([f"Line {i}: Some error details here" for i in range(50)])
                await tg.send(long_message, use_summary=True, prefix="Error Summary: ")
                
                # Test error alert
                await tg.send_error_alert("Test error message", "Unit test context")
                
                logger.info("All tests completed.")
                
            except Exception as e:
                logger.error(f"Test failed: {e}")

    # Alternative simple test using utility function
    async def simple_test():
        """Simple test using utility function."""
        try:
            await send_telegram_message("Simple test message!")
            logger.info("Simple test completed.")
        except Exception as e:
            logger.error(f"Simple test failed: {e}")

    # Graceful shutdown helper
    async def shutdown_gracefully():
        """Helper to shutdown with proper cleanup."""
        try:
            # Cancel all pending tasks
            loop = asyncio.get_running_loop()
            pending = asyncio.all_tasks(loop)
            pending.discard(asyncio.current_task())
            
            if pending:
                logger.info(f"Cancelling {len(pending)} pending tasks...")
                for task in pending:
                    task.cancel()
                
                await asyncio.gather(*pending, return_exceptions=True)
                logger.info("All tasks cancelled.")
                
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    # Run the test with proper cleanup
    async def run_tests():
        try:
            print("Running comprehensive test...")
            await test_bot()
            
            print("\nRunning simple test...")
            await simple_test()
            
        except KeyboardInterrupt:
            logger.info("Tests interrupted by user.")
        except Exception as e:
            logger.error(f"Test suite failed: {e}")
        finally:
            logger.info("Performing graceful shutdown...")
            await shutdown_gracefully()
            await asyncio.sleep(0.5)  # Final cleanup delay

    # Run the test
    asyncio.run(run_tests())