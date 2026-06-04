import logging
from datetime import datetime
import os
from rich.logging import RichHandler


def setup_logger() -> None:
    os.makedirs("logs", exist_ok=True)
    log_file_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = f"logs/{log_file_time}.txt"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            # format="%(asctime)s | %(lineno)d | %(funcName)s | %(levelname)s | %(message)s",
            fmt="%(asctime)s | %(funcName)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d_%H-%M-%S",
        )
    )

    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=True,
    )
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


setup_logger()

logger = logging.getLogger(__name__)
