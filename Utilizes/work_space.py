#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @Copyright (c) 2023 Baidu.com, Inc. All Rights Reserved
# @Time    : 2023/7/16 22:58
# @Author  : Liu Tianyuan (liutianyuan02@baidu.com)
# @File    : workspace.py
# @Description    : ******
"""

import os
import sys
import logging

class WorkSpace(object):
    """
    WorkSpace: Create a work space for a specific work.
    """
    def __init__(self, work_name, base_path=None):

        self.work_name = work_name
        self.basic_path = os.path.join(base_path, 'work', self.work_name)
        self.train_path = os.path.join(self.basic_path, 'train')
        self.valid_path = os.path.join(self.basic_path, 'valid')
        self.logger_file = os.path.join(self.basic_path, self.work_name + '.log')

        # create all path of work space
        self._create_path()
        self.logger = TextLogger(self.logger_file)
        self.logger.info('Create work space: {}'.format(self.basic_path))

    def _create_path(self):
        """
            Create a work space for a specific work.
        """
        for key in self.__dict__:
            if key.endswith('_path'):
                isCreated = os.path.exists(self.__dict__[key])
                if not isCreated:
                    os.makedirs(self.__dict__[key])



class TextLogger(object):
    """
    log文件记录所有打印结果
    """

    def __init__(self, filename, level=logging.INFO, stream=sys.stdout):
        self.terminal = stream
        # self.log = open(filename, 'a')

        formatter = logging.Formatter("%(levelname)s: %(asctime)s:   %(message)s",
                                      "%m-%d %H:%M:%S")
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # logger = logging.getLogger()
        # logger.setLevel(level)
        handler = logging.FileHandler(filename)
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    @staticmethod
    def info(message: str):
        # info级别的日志，绿色
        logging.info("\033[0;32m" + message + "\033[0m")

    @staticmethod
    def warning(message: str):
        # warning级别的日志，黄色
        logging.warning("\033[0;33m" + message + "\033[0m")

    @staticmethod
    def important(message: str):
        # 重要信息的日志，红色加下划线
        logging.info("\033[4;31m" + message + "\033[0m")

    @staticmethod
    def conclusion(message: str):
        # 结论级别的日志，紫红色加下划线
        logging.info("\033[4;35m" + message + "\033[0m")

    @staticmethod
    def error(message: str):
        # error级别的日志，红色
        logging.error("\033[0;31m" + "-" * 120 + '\n| ' + message + "\033[0m" + "\n" + "└" + "-" * 150)

    @staticmethod
    def debug(message: str):
        # debug级别的日志，灰色
        logging.debug("\033[0;37m" + message + "\033[0m")

    @staticmethod
    def write(message):
        """
        文本输出记录
        """
        logging.info(message)

    def flush(self):
        """
        通过
        """
        pass


if __name__ == "__main__":

    workspace = WorkSpace("test", base_path=os.getcwd())


