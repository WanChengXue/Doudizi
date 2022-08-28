"""
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-08-14 21:28:45
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-08-16 20:38:48
FilePath: /RLFramework/Worker/Agent/FC_agent.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""
import torch

import os
import sys

current_path = os.path.abspath(__file__)
root_path = "/".join(current_path.split("/")[:-3])
sys.path.append(root_path)


from Utils.model_utils import deserialize_model, create_model


class Agent:
    def __init__(self, config):
        # ------------- 创建网络，包括actor网络和critic网络 --------------------
        self.agent_config = config
        # ----------- 构建智能体 ----------
        self.construct_model()

    def construct_model(self):
        # --------- 如果训练的模型是landlord，则直接加载一个farmer model -------
        if "policy" in self.agent_config:
            self.policy_net = create_model(self.agent_config["policy"])

    def synchronize_model(self, model_path):
        # ------------ 这个函数用来同步本地策略模型 ——-----------
        for key in model_path.keys():
            if key == "policy":
                deserialize_model(self.policy_net, model_path["policy"])
            elif key == "critic":
                deserialize_model(self.critic_net, model_path["critic"])
            elif key == "centralized_critic":
                deserialize_model(
                    self.centralized_critic_net, model_path["centralized_critic"]
                )
            else:
                pass

    def compute_action_training_mode(self, obs):
        with torch.no_grad():
            action, mask = self.policy_net(**obs)
        # ------ 获得一个mask向量 ----
        action_dict = dict()
        action_dict["action"] = action
        action_dict["mask"] = mask
        return action_dict

    def compute_action_eval_mode(self, obs):
        with torch.no_grad():
            Q_value = self.policy_net(**obs, return_value=True)
            action = torch.argmax(Q_value, dim=0)[0]
        return action
