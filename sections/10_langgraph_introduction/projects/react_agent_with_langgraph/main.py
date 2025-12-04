from dotenv import load_dotenv
import os
import asyncio
from logger import log_success, log_error, log_warning, log_header

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not set in environment or .env file")

def main():
    log_success("Hello from react-agent-with-langgraph!")


if __name__ == "__main__":
    main()
