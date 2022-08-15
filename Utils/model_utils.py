'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-08-13 14:55:26
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-08-15 22:30:04
FilePath: /RLFramework/Utils/model_utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from collections import OrderedDict
import torch
import importlib
import time
import os
import glob


def serialize_model(model_path_prefix, model_url_prefix, net_dict, cache_size, logger):
    # --------- 这个函数是将模型保存到本地,并且将模型的路径字典返回 --------------
    '''
        model_path_prefix表示的是模型路径的前缀
        model_url_prefix表示的是learner发布的模型的路径,发送到config_server
        如果是单机跑，其实上面两个路径是一样的，但是分开的化，model_url需要由这个机器的ip地址组成
        net_dict表示的是模型构成的字典
        cache_size表示的是模型最大缓存数目
    '''
    # ---------- model_path_prefix是Exp/Model_pool/default_policy ------------
    # ----------- model_url_paht 是 http://ip:port ------------
    # ------------ 传入的必然是字典形式，如果是共享参数，则net_dict['policy']['default'] = model
    url_dict = dict()
    timestamp = str(time.time())
    for agent_name in net_dict.keys():
        if isinstance(net_dict[agent_name], dict):
            sub_url_dict = dict()
            for model_type in net_dict[agent_name].keys():
                model_save_path = model_path_prefix + '/' + agent_name +  '_' + model_type + '_' + timestamp
                model_url_path = model_url_prefix + '/' + agent_name +  '_' +  model_type  + '_' + timestamp
                sub_url_dict[model_type] = model_url_path
                torch.save(net_dict[agent_name][model_type].state_dict(), model_save_path)
                remove_old_version_model(model_path_prefix+'/'+agent_name+  '_' +model_type+'*', cache_size, logger)
            url_dict[agent_name] = sub_url_dict
        else:
            model_save_path = model_path_prefix + '/' + agent_name + '_' + timestamp
            model_url_path = model_url_prefix + '/' + agent_name + '_' + timestamp
            torch.save(net_dict[model_type].state_dict(), model_save_path)
            remove_old_version_model(model_path_prefix+'/'+agent_name+'*', cache_size, logger)
            url_dict[agent_name] = model_url_path
    return url_dict


def remove_prefix(state_dict, prefix):
    # -------- 这个函数是用来移除前缀prefix -------
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return OrderedDict({f(key): value  for key, value in state_dict.items()})

def deserialize_model(model_object, model_parameter_dict, device='cpu'):
    # ----------- 这个函数是从内存中载入模型，推断的时候，模型放在CPU上 --------
    # ----------- 这个函数接受的两个参数，第一个是pytorch模型，第二个是对应模型的参数 ----------
    saved_state_dict = torch.load(model_parameter_dict, map_location=device)
    # -------- 使用DDP包装了一层的模型，保存到本地的时候都会包一个prefix前缀 --------
    new_state_dict = remove_prefix(saved_state_dict, 'module.')
    model_object.load_state_dict(new_state_dict)


def create_model(model_config):
    model_name = model_config['model_name']
    model_fn = getattr(importlib.import_module("Model.{}".format(model_name)), 'init_{}_{}_net'.format(model_config['model_type'], model_config['agent_name']))
    return model_fn(model_config)
    

def remove_old_version_model(model_prefix, cache_size, logger):
    # -------------- 这个函数是用来移除多余的模型 -------------
    # ----------- 首先返回这个model_prefix路径下面所有的文件构成的list -------
    model_files = glob.glob(model_prefix)
    if len(model_files)> cache_size:
        sorted_model_files = sorted(model_files)
        old_model = sorted_model_files[0]
        os.remove(old_model)
        logger.info('---------------- 移除多余的模型: {} ---------------'.format(old_model))
