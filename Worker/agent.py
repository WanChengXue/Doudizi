import os
import sys
import importlib

current_path = os.path.abspath(__file__)
root_path = "/".join(current_path.split("/")[:-2])
sys.path.append(root_path)
import Worker


def get_agent(agent_file):
    agent_obj = getattr(
        importlib.import_module("Worker.Agent.{}".format(agent_file)), "Agent"
    )
    return agent_obj
