import yaml
import os
from yaml import Loader
from copy import deepcopy

from Utils.utils import create_folder


def load_yaml(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.load(f, Loader=Loader)
    return config_dict


def _read_model_from_model_pool(
    root_path, model_path, config_dict, imitation_policy=None
):
    # 这样写的原因是，使用RL算法加载预训练模型的时候，需要从pretrained model中进行载入
    if imitation_policy is None:
        model_pool_path = os.path.join(
            root_path, model_path + config_dict["policy_name"]
        )
    else:
        model_pool_path = os.path.join(root_path, model_path + imitation_policy)

    for agent_name in config_dict["env"]["agent_name_list"]:
        if agent_name in config_dict["env"]["trained_agent_name_list"]:
            if config_dict["policy_config"].get("parameter_sharing", True):
                # ---------- 如果参数共享,就只要读取一个就好了 --------
                model_pool_file = sorted(
                    [
                        file_name
                        for file_name in os.listdir(model_pool_path)
                        if "policy" in file_name
                    ]
                )
            else:
                model_pool_file = sorted(
                    [
                        file_name
                        for file_name in os.listdir(model_pool_path)
                        if agent_name + "_policy" in file_name
                    ]
                )
            config_dict["policy_config"]["agent"][agent_name]["policy"][
                "model_path"
            ] = os.path.join(model_pool_path, model_pool_file[-1])


def _double_q_network_repeat_config(config_dict):
    # --------- 这个函数是说,如果使用了double q网络就将critic的配置复制一份,然后关键字为double_xx --------
    for agent_name in config_dict["env"]["agent_name_list"]:
        if "critic" in config_dict["policy_config"]["agent"][agent_name].keys():
            double_critic_config = deepcopy(
                config_dict["policy_config"]["agent"][agent_name]["critic"]
            )
            config_dict["policy_config"]["agent"][agent_name][
                "double_critic"
            ] = double_critic_config
    if "centralized_critic" in config_dict["policy_config"]["agent"].keys():
        double_critic_config = deepcopy(
            config_dict["policy_config"]["agent"]["centralized_critic"]
        )
        config_dict["policy_config"]["agent"][
            "double_centralized_critic"
        ] = double_critic_config


def _get_state_and_action_dim_from_policy_config(config_dict):
    # ----------- 如果说采用了中心的critic ------------
    if "state_dim" in config_dict["policy_config"]["agent"]["centralized_critic"]:
        return
    else:
        state_dim_list = []
        action_dim_list = []
        for agent_name in config_dict["env"]["agent_name_list"]:
            state_dim_list.append(
                config_dict["policy_config"]["agent"][agent_name]["policy"]["state_dim"]
            )
            action_dim_list.append(
                config_dict["policy_config"]["agent"][agent_name]["policy"][
                    "action_dim"
                ]
            )
            config_dict["policy_config"]["agent"]["centralized_critic"][
                "state_dim"
            ] = sum(state_dim_list)
            config_dict["policy_config"]["agent"]["centralized_critic"][
                "action_dim"
            ] = sum(action_dim_list)


def _repeat_agent_config(config_dict):
    # --------- 这个是在使用了参数共享的时候, 需要复制多份参数  ------------
    if config_dict["policy_config"]["parameter_sharing"]:
        agent_dict = dict()
        for agent_name in config_dict["env"]["agent_name_list"]:
            agent_dict[agent_name] = deepcopy(config_dict["policy"]["agent"]["default"])
        config_dict["policy"]["agent"] = agent_dict


def parse_config(config_file_path, parser_obj="learner"):
    function_path = os.path.abspath(__file__)
    # ------ 这个就是到了Pretrained_model这一层路径下面 ----- ~/Desktop/pretrained_model
    root_path = "/".join(function_path.split("/")[:-2])
    config_dict = load_yaml(config_file_path)
    _repeat_agent_config(config_dict)
    if config_dict["policy_config"].get("eval_mode", False):
        _read_model_from_model_pool(
            root_path,
            config_dict["policy_config"]["pretrained_model_path"],
            config_dict,
        )
        # ------------   在eval模式下面，需要创建一个文件夹，然后将采样结果放到里面去 -----------
        result_save_path = os.path.join(
            root_path, "Exp/Result/Evaluate/{}".format(config_dict["policy_name"])
        )
        create_folder(result_save_path)
        config_dict["policy_config"]["result_save_path"] = result_save_path
        return config_dict
    if config_dict["policy_config"].get("double_dqn", False):
        # ------------ 如果说使用了double Q net,则需要将critic的配置重复一次 ------------
        _double_q_network_repeat_config(config_dict)
    # ------------ 如果说使用的RL相关的算法，就需要设置critic的state和action dim ------
    if config_dict["policy_config"]["training_type"] != "supervised_learning":
        if "centralized_critic" in config_dict["policy_config"]["agent"]:
            _get_state_and_action_dim_from_policy_config(config_dict)
            config_dict["policy_config"]["use_centralized_critic"] = True
        else:
            config_dict["policy_config"]["use_centralized_critic"] = False

    if config_dict.get("load_data_from_model_pool", False) and parser_obj == "learner":
        # ---------- 在非eval模式下，并且只有learner才需要读取模型，需要从预训练模型路径下面进行读取 --------
        config_dict = _read_model_from_model_pool(
            root_path,
            config_dict["policy_config"]["pretrained_model_path"],
            config_dict,
            config_dict["policy_config"]["imitation_policy_name"],
        )

    if "main_server_ip" in config_dict:
        config_dict["log_server_address"] = config_dict["main_server_ip"]
        config_dict["config_server_address"] = config_dict["main_server_ip"]
    config_dict["log_dir"] = os.path.join(
        config_dict["log_dir"], config_dict["policy_name"]
    )
    create_folder(config_dict["log_dir"])
    # ----------------- 覆盖掉原始的值 ----------------------------------------------------------------------
    # 使用单机多卡去运行
    main_server_ip = config_dict["main_server_ip"]
    policy_config = config_dict["policy_config"]
    # ddp相关参数
    ddp_port = policy_config["ddp_port"]
    policy_config["ddp_root_address"] = "tcp://{}:{}".format(main_server_ip, ddp_port)
    # --------------- 这个地方对config_dict中的learner部分进行修改，主要是将env中的一些参数复制过来 ---------------
    # ---- 处理一下plasma的保存位置，改成绝对位置,将父文件夹创建出来，然后client连接一定是文件 ----
    policy_config["plasma_server_location"] = (
        root_path + "/" + policy_config["plasma_server_location"]
    )
    create_folder(policy_config["plasma_server_location"])
    policy_config["plasma_server_location"] = (
        policy_config["plasma_server_location"] + "/" + config_dict["policy_name"]
    )
    # --------- 构建模型发布的url --------------------
    http_server_ip = "http://{}:{}".format(
        config_dict["config_server_address"], config_dict["config_server_http_port"]
    )
    policy_config["model_pool_path"] = os.path.join(
        policy_config["model_pool_path"], config_dict["policy_name"]
    )
    policy_config["saved_model_path"] = os.path.join(
        policy_config["saved_model_path"], config_dict["policy_name"]
    )
    # ----------- 将model_pool_path和saved_model_path直接构建成绝对路径 ----------------
    abs_model_pool_path = os.path.join(root_path, policy_config["model_pool_path"])
    create_folder(abs_model_pool_path)
    create_folder(policy_config["saved_model_path"])
    policy_config["model_url"] = http_server_ip
    config_dict["policy_config"] = policy_config
    # ------------- 修改一下tensorboard的保存路劲 ----------
    policy_config["tensorboard_folder"] = os.path.join(
        config_dict["log_dir"], policy_config["tensorboard_folder"]
    )
    # ------------- 最后就是说，worker需要从configserver上面下载新模型，就在本地创建一个文件夹出来 -------------
    create_folder("./Worker/Download_model")
    create_folder(config_dict["data_saved_folder"])
    # ----------- 保存yaml文件 -----------
    yaml_saved_path = os.path.join(
        config_dict["log_dir"], "{}.yaml".format(config_dict["policy_name"])
    )
    if not os.path.exists("config.yaml"):
        # ------ 如果说配置文件不存在，就直接保存到本地 -----
        with open(yaml_saved_path, "w", encoding="utf8") as f:
            yaml.dump(config_dict, f)
    return config_dict
