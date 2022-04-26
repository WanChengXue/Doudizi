import torch


def soft_update(current_network, tart_network, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(tart_network.parameters(), current_network.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau)

def alpha_sync(current_model, target_model, alpha):
    """
    Blend params of target net with params from the model
    :param alpha:
    """
    assert isinstance(alpha, float)
    assert 0.0 < alpha <= 1.0
    state = current_model.state_dict()
    tgt_state = target_model.state_dict()
    for k, v in state.items():
        tgt_key = k.split('module.', 1)[-1] if k.startswith('module.') else k
        tgt_state[tgt_key] = tgt_state[tgt_key] * (1 - alpha) + alpha * v
    target_model.load_state_dict(tgt_state)

def huber_loss(a,b, delta=1.0):
    gap = a-b
    flag_matrix = (torch.abs(gap)<= delta).float()
    mse_loss = 0.5 * gap ** 2
    other_branch = delta * (torch.abs(gap) - 0.5*delta)
    return torch.sum(flag_matrix * mse_loss + (1-flag_matrix) * other_branch, -1)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

def reversed_dict(action_dict): 
    '''
    传入的action_dcit = {
        'cycle' : {'agent_0': tensor, ...},
        'splits' : {'agent_0': tensor, ...}
    }
    
    '''
    reversed_action_dict = dict()
    # ------- 传入的dict是rosea dict，需要拼接得到一个tensor --------
    action_keys_list = list(action_dict.keys())
    agent_keys_list = list(action_dict[action_keys_list[0]].keys()) 
    for agent_key in agent_keys_list:
        agent_action_list = []
        for action_key in action_keys_list:
            agent_action_list.append(action_dict[action_key][agent_key])
        reversed_action_dict[agent_key] = torch.cat(agent_action_list, -1)
    return reversed_action_dict
            

