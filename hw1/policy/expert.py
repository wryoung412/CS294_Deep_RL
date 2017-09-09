import importlib

def get_policy(env_name):
    policy_module = importlib.import_module("experts." + env_name)
    _, policy = policy_module.get_env_and_policy()
    return policy
