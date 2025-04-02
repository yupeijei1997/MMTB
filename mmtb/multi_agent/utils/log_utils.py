import logging


handler = logging.StreamHandler()  # 输出到命令行
handler.flush()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[handler]
)
logger = logging.getLogger()
