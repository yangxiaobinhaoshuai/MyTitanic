import logging
from datetime import datetime
import os
from rich.logging import RichHandler

now = datetime.now()

print(type(now))
print(now)

format_now = now.strftime("%Y-%m-%d_%H:%M:%S")

print(type(format_now))
print(format_now)

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(lineno)d | %(funcName)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(f"logs/{format_now}.txt"),
        RichHandler(rich_tracebacks=True),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)
