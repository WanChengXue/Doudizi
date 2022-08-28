import logging
import os
import shutil
import sys


def create_folder(folder_path, delete_origin=False):
    # 这个函数的作用是,当文件夹存在就删除,然后重新创建一个新的
    if not os.path.exists(folder_path):
        # shutil.rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)
    else:
        if delete_origin:
            shutil.rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)


def check_folder_exist(folder_path):
    # 这个函数是检查传入的文件夹路径中是不是空的,如果是空的就创建,不是就什么也不做
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return False
    else:
        return True


def setup_logger(name, log_folder_path, level=logging.DEBUG):
    create_folder(log_folder_path)
    log_file = log_folder_path / "log"
    handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    formatter = logging.Formatter(
        "%(asctime)s,%(msecs)d,%(levelname)s,%(name)s::%(message)s"
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def generate_plasma_id(machine_index, device_index, data_server_index):
    # 这个函数是用来生成独一无二的plasma id，必须是长度为20的bytes list
    # 组合规则，根据这个dataserver的机器id，设备的索引，进程的索引三个部分构成
    plasma_id = (
        "plasma_id"
        + str(machine_index) * 3
        + str(device_index) * 4
        + str(data_server_index) * 4
    )
    if len(plasma_id) > 20:
        plasma_id = plasma_id[:20]
    return bytes(plasma_id, encoding="utf-8")


def generate_plasma_id_for_PEB(machine_index, device_index, data_server_index):
    plasma_id = (
        "plasma_id_PEB"
        + str(machine_index) * 2
        + str(device_index) * 2
        + str(data_server_index) * 3
    )
    if len(plasma_id) > 20:
        plasma_id = plasma_id[:20]
    return bytes(plasma_id, encoding="utf-8")
