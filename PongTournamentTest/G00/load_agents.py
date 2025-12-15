import os
from stable_baselines3 import A2C
from G00.model import SymmetricActorCriticPolicy

"""
# Path to the agent's models
agent_left_name = "a2c_pong_marl_selfplay_continued.zip"
agent_right_name = "a2c_pong_marl_selfplay_continued.zip"


def load_right_agent():
    '''
    Loads and returns the right agent model.
    Returns:
        agent: The loaded left agent model.
    '''
    agent = A2C.load(os.path.join(os.path.dirname(__file__), agent_right_name))

    return agent


def load_left_agent():
    '''
    Loads and returns the left agent model.
    Returns:
        agent: The loaded left agent model.
    '''
    agent = A2C.load(os.path.join(os.path.dirname(__file__), agent_left_name))

    return agent
    """

def load_left_agent():
    from stable_baselines3 import A2C
    # Import the custom policy definition if it's in a separate file, 
    # or ensure it is available in the scope.
                    
    model = A2C.load(os.path.join(os.path.dirname(__file__), "left_specialist_4200000_steps.zip"),
                custom_objects={'policy_class': SymmetricActorCriticPolicy})
                            
    model.policy.force_flip = True
                                                
    return model

def load_right_agent():
    model = A2C.load(os.path.join(os.path.dirname(__file__), "final.zip"),
                     custom_objects={'policy_class': SymmetricActorCriticPolicy})
    model.policy.force_flip = False # Default
    return model
