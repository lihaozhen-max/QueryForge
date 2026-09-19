import asyncio
import sys

from loguru import logger

from app.conf.app_config import app_config
from app.core.context import set_request_id, get_request_id, reset_request_id

log_format = (
    "<red>{time:YYYY-MM-DD HH:mm:ss.SSS}</red> | "  # 绿色显示日志时间（精确到毫秒）
    "<level>{level: <8}</level> | "  # 按级别颜色显示日志级别（左对齐，占8个字符）
    "<magenta>request_id - {extra[request_id]}</magenta> | "  # 品红色显示request_id（从日志extra中获取）
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "  # 青色显示日志所在文件、函数、行号
    "<level>{message}</level>"  # 按级别颜色显示日志正文
)

logger.remove()

def inject_request_id(record):
    # print('-------')
    record["extra"]["request_id"] = get_request_id()

logger = logger.patch(inject_request_id)

logger.add(
    sink=sys.stdout,
    level=app_config.logging.console.level,
    format=log_format,
)

logger.add(
    sink='log/app.log',
    level=app_config.logging.file.level,
    format=log_format,
    rotation=app_config.logging.file.rotation,
    retention=app_config.logging.file.retention,
)









if __name__ == '__main__':
    # set_request_id("1111")
    logger.trace("这是 TRACE 级别的调试信息")  # 不会输出到控制台（控制台是 INFO），但会输出到文件
    logger.debug("这是 DEBUG 级别的调试信息")  # 同上
    logger.info("服务启动成功")  # 控制台+文件都输出
    logger.success("数据同步完成")  # Loguru 独有级别
    logger.warning("内存使用率超过 80%")
    logger.error("接口调用失败：超时")
    logger.critical("数据库连接中断，服务停止")


    async def req1():
        token = set_request_id("1111")
        await asyncio.sleep(1)
        logger.info(f'-2-req1 req_id={get_request_id()}')
        await asyncio.sleep(1)
        reset_request_id(token)
        logger.info(f'-3-req1 req_id={get_request_id()}')

    async def req2():
        token = set_request_id('2222')
        await asyncio.sleep(1)
        logger.info(f'-2-req2 req_id={get_request_id()}')
        await asyncio.sleep(1)
        reset_request_id(token)
        logger.info(f'-3-req2 req_id={get_request_id()}')

    async def test():
        await asyncio.gather(req1(), req2())

    asyncio.run(test())







