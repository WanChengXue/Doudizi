import torch
import torch.nn as nn

def get_cls():
    return SLTrainer


class SLTrainer:
    def __init__(self, model, optimizer, scheduler, policy_config):
        self.policy_config = policy_config
        self.regulization = float(self.policy_config['regulization'])
        # ---------- 需要区分，如果是多个异构的神经网络，如何处理的问题 --------
        self.optimizer = optimizer['policy']
        self.scheduler = scheduler['policy']
        self.model = model['policy']
        self.loss_fn = nn.L1Loss(reduction='none')
        self.max_grad_norm = self.policy_config['max_grad_norm']
        self.parameter_sharing = self.policy_config.get('parameter_sharing',True)
        self.agent_name_list = model['policy'].keys()

    def huber_loss(self, a,b, delta=1.0):
        gap = a-b
        flag_matrix = (torch.abs(gap)<= delta).float()
        mse_loss = 0.5 * gap ** 2
        other_branch = delta * (torch.abs(gap) - 0.5*delta)
        return torch.sum(flag_matrix * mse_loss + (1-flag_matrix) * other_branch, -1)

    def step(self, training_data):
        state = training_data['current_state']
        label = training_data['action']
        splits_weight = training_data['splits']
        cycle_weight = training_data['cycle']
        info_dict = dict()
        for index, key in enumerate(state.keys()):
            single_tls_state = state[key]
            single_tls_splits = label['splits'][key]
            single_tls_cycle = label['cycle'][key]
            # 将splits和cycle进行拼接
            single_tls_label = torch.cat([single_tls_splits, single_tls_cycle], -1)
            single_tls_data_weight = torch.cat([splits_weight[key], cycle_weight[key]], -1)
            if self.parameter_sharing:
                algo_key = 'default'
            else:
                algo_key = 'agent_{}'.format(index)
            network_output = self.model[algo_key](single_tls_state)
            matrix_loss = self.loss_fn(network_output, single_tls_label)
            action_mean_loss = torch.sum(single_tls_data_weight * matrix_loss, -1)
            loss = action_mean_loss.mean()
            # ----------------- L2 norm ---------------
            layer_norm_list = []
            # --------- 记录权重矩阵的最大值和最小值 ----------
            layer_max_weights = {}
            layer_min_weights = {}
            for name, value in self.model[algo_key].named_parameters():
                layer_norm_list.append(torch.sum(torch.pow(value, 2)))
                layer_max_weights[name] = torch.max(value).item()
                layer_min_weights[name] = torch.min(value).item()

            l2_norm_loss = sum(layer_norm_list)
            total_loss = loss + l2_norm_loss * 0.5* self.regulization  
            self.optimizer[algo_key].zero_grad()
            total_loss.backward()
            # ------------- 进行梯度clip操作 -----------------
            nn.utils.clip_grad_norm_(self.model[algo_key].parameters(), self.max_grad_norm)
            self.optimizer[algo_key].step()
            self.scheduler[algo_key].step()
            # ------------ 记录梯度信息 ---------------
            layer_max_grads = {}
            layer_min_grads = {}
            for name, value in self.model[algo_key].named_parameters():
                layer_max_grads[name] = torch.max(value.grad).item()
                layer_min_grads[name] = torch.min(value.grad).item()
            info = {
                'Loss/Predict_loss': loss.item(),
                'Loss/L2Norm_loss': l2_norm_loss.item(),
                'Loss/Total_loss': total_loss.item(),
                'Model/Layer_max_weights': layer_max_weights,
                'Model/Lyaer_min_weights': layer_min_weights,
                'Model/Layer_max_grads': layer_max_grads,
                'Model/Layer_min_grads': layer_min_grads 
            }
            info_dict[algo_key] = info
        return info_dict