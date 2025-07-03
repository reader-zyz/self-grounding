# log.py
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

def setup_logging(log_dir="logs"):
    """配置按日期滚动的日志系统
    
    Args:
        log_dir (str): 日志存储目录，默认'logs'
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成带日期的日志文件名
    log_filename = f"app_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # 日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    formatter = logging.Formatter(log_format)
    
    # 文件处理器（按天滚动）
    file_handler = TimedRotatingFileHandler(
        filename=log_path,
        when="midnight",  # 每天新建文件
        backupCount=7,    # 保留7天日志
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.suffix = "%Y%m%d.log"  # 备份文件后缀
    
    # 配置根日志器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # 可选：控制台输出（调试用）
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 返回配置好的日志器（非必须）
    return logger

# # 初始化全局日志（导入log.py时自动执行）
# setup_logging(log_dir="my_logs")