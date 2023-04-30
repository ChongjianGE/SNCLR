import os
from loguru import logger

def setup_logger(save_dir, gpu=0, rank=0, filename="log.txt", mode="a"):
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    if gpu != 0 or rank != 0:
        logger.remove()
    else:
        logger.add(
            save_file, format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}", filter="", level="INFO", enqueue=True)

    return logger
