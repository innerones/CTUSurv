import logging
import time
import os


class Logger:
    def __init__(self, logger_name):
        self.logger = logging.getLogger(logger_name)

        self.logger.setLevel(logging.DEBUG)
        basedir = os.path.abspath(os.path.dirname(__file__))
        log_path = os.path.join(basedir, 'logs', time.strftime("%F"))  # 日志根目录 ../logs/yyyy-mm-dd/

        if not os.path.exists(log_path):
            os.mkdir(log_path)

        log_name = os.path.join(log_path, 'out.log')
        fh = logging.FileHandler(log_name, encoding='utf-8', mode='a')  # 指定utf-8格式编码，避免输出的日志文本乱码
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def get_log(self):
        return self.logger

    def shutdown(self):
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)