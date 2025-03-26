import os.path as osp
import highway_safety_gym
from safe_rl.runner import Runner
from safe_rl.util.run_util import load_config, setup_eval_configs
from safe_rl.util.logger import setup_logger_kwargs, EpochLogger
import gym
from gym import Wrapper
from gym.envs.registration import register
import numpy as np
import json

# Import the enhanced environment directly
from highway_safety_gym.envs.enhanced_highway_env import EnhancedHighwaySafetyEnv

# Import the debug CVPO class
from debug_cvpo import DebugCVPO

# Patch the Runner.POLICY_LIB to use the debug CVPO class
def patch_policy_lib():
    """Patch the POLICY_LIB to use the debug CVPO class"""
    Runner.POLICY_LIB['cvpo'] = (DebugCVPO, Runner.POLICY_LIB['cvpo'][1], Runner.POLICY_LIB['cvpo'][2])
    print("Patched Runner.POLICY_LIB to use DebugCVPO")

# A wrapper that handles API format conversion
class AdaptiveWrapper(Wrapper):
    """
    A wrapper that always returns the old Gym API format (obs, reward, done, info)
    for compatibility with the worker code.
    """
    def __init__(self, env):
        super().__init__(env)
        print(f"AdaptiveWrapper always returns old API format (external)")
        
    def step(self, action):
        """
        Always return the old API format (obs, reward, done, info) for compatibility
        with the worker code.
        """
        try:
            # Try calling the environment with the action
            result = self.env.step(action)
            
            # Check what format the result is in
            if len(result) == 5:  # New API (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
                return obs, reward, done, info
            else:  # Old API (obs, reward, done, info)
                return result
        except Exception as e:
            print(f"Error in AdaptiveWrapper.step: {e}")
            # If there was an error, try a different approach
            try:
                # Try to get the result in new API format
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                return obs, reward, done, info
            except:
                # Last resort - try old API
                return self.env.step(action)
    
    def reset(self, **kwargs):
        """
        Reset the environment and return just the observation for old API compatibility.
        """
        try:
            # Try calling reset and see what format it returns
            result = self.env.reset(**kwargs)
            if isinstance(result, tuple) and len(result) == 2:  # New API (obs, info)
                # Return just the observation for old API
                return result[0]
            else:  # Assume it's just the observation (old API)
                return result
        except Exception as e:
            print(f"Error in AdaptiveWrapper.reset: {e}")
            # If error, try simplest approach
            try:
                return self.env.reset()
            except:
                # Last resort
                obs = self.env.reset()
                if isinstance(obs, tuple):
                    return obs[0]
                return obs

CONFIG_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "config")

class SafetyGymRunner(Runner):
    def __init__(self, **kwargs):
        # Store kwargs to use in _train_mode_init
        self.kwargs = kwargs
        super().__init__(**kwargs)
        
    def _train_mode_init(self, env, seed, exp_name, policy, timeout_steps, data_dir, **kwarg):
        # Create environment - handle enhanced environment specially
        if env == 'EnhancedSafeHighway-v0':
            # Create the enhanced environment directly to avoid Gym wrappers
            self.env = EnhancedHighwaySafetyEnv(cost_limit=10)
            
            # Add a spec attribute with an id that contains "Safe"
            class DummySpec:
                def __init__(self, id):
                    self.id = id
            
            self.env.spec = DummySpec(id="SafeEnhancedHighway-v0")  # Note "Safe" prefix
            print(f"Created EnhancedHighwaySafetyEnv directly with dummy spec")
        else:
            # Create via gym.make for other environments
            self.env = gym.make(env)
            self.env = AdaptiveWrapper(self.env)
            print(f"Created environment via gym.make: {env}")
        
        # Use try/except for seeding in case the method doesn't exist
        try:
            self.env.seed(seed)
        except AttributeError:
            print("Warning: Environment does not have a seed method")
            # Try to set random seed directly
            np.random.seed(seed)
        
        # Debug: Check the observation
        initial_obs = self.env.reset()
        print(f"Initial observation: {initial_obs}")
        print(f"Observation type: {type(initial_obs)}")
        if isinstance(initial_obs, np.ndarray):
            print(f"Observation shape: {initial_obs.shape}")
        
        # Continue with the original initialization
        self.timeout_steps = getattr(self.env, '_max_episode_steps', 100) if timeout_steps == -1 else timeout_steps
        
        # The rest is the same as the original method
        logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir=data_dir)
        self.logger = EpochLogger(**logger_kwargs)
        
        # Create a JSON-serializable config
        config = {
            'env': env,
            'seed': seed,
            'exp_name': exp_name,
            'policy': policy,
            'timeout_steps': timeout_steps,
            'data_dir': data_dir
        }
        
        # Add kwargs safely (skip non-serializable items)
        for key, value in self.kwargs.items():
            if key != 'self' and key != 'logger_kwargs' and key != 'kwarg':
                try:
                    # Test if the value is JSON serializable
                    json.dumps({key: value})
                    config[key] = value
                except (TypeError, OverflowError):
                    # Skip non-serializable values
                    print(f"Skipping non-serializable config value: {key}")
        
        # Add policy config
        if policy in kwarg:
            config[policy] = kwarg[policy]
            
        self.logger.save_config(config)
        
        # Init policy
        self.policy_config = kwarg[policy]
        self.policy_config["timeout_steps"] = self.timeout_steps
        policy_cls, self.on_policy, worker_cls = self.POLICY_LIB[policy.lower()]
        self.policy = policy_cls(self.env, self.logger, **self.policy_config)
        
        if self.pretrain_dir is not None:
            model_path, _, _, _, _ = setup_eval_configs(self.pretrain_dir)
            self.policy.load_model(model_path)
        
        self.steps_per_epoch = self.policy_config[
            "steps_per_epoch"] if "steps_per_epoch" in self.policy_config else 1
        self.worker_config = self.policy_config["worker_config"]
        self.worker = worker_cls(self.env,
                                 self.policy,
                                 self.logger,
                                 timeout_steps=self.timeout_steps,
                                 **self.worker_config)
    
    def eval(self, epochs=10, sleep=0.01, render=True):
        '''
        Overwrite the eval function since the rendering for 
        bullet safety gym is different from other gym envs
        '''
        if render:
            self.env.render()
        super().eval(epochs, sleep, False)

EXP_NAME_KEYS = {"epochs": "epoch"}
DATA_DIR_KEYS = {"cost_limit": "cost"}

def gen_exp_name(config: dict, suffix=None):
    suffix = "" if suffix is None else "_" + suffix
    name = config["policy"]
    for k in EXP_NAME_KEYS:
        name += '_' + EXP_NAME_KEYS[k] + '_' + str(config[k])
    return name + suffix

def gen_data_dir_name(config: dict):
    name = config["env"]
    for k in DATA_DIR_KEYS:
        name += '_' + DATA_DIR_KEYS[k] + '_' + str(config[k])
    return name

if __name__ == '__main__':
    # Patch the POLICY_LIB before creating the runner
    patch_policy_lib()
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', type=str, default='Safexp-PointButton1-v0')
    parser.add_argument('--policy', '-p', type=str, default='cvpo')
    parser.add_argument('--pretrain_dir', '-pre', type=str, default=None)
    parser.add_argument('--load_dir', '-d', type=str, default=None)
    parser.add_argument('--mode', '-m', type=str, default='train')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--suffix', '--id', type=str, default=None)
    parser.add_argument('--no_render', action="store_true")
    parser.add_argument('--sleep', type=float, default=0.003)
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    args = parser.parse_args()
    args_dict = vars(args)
    
    # Check what environments are registered
    from gym.envs.registration import registry
    print("Available environments:")
    for env_id in sorted(registry.keys()):
        if "highway" in env_id or "Safe" in env_id:
            print(f"  - {env_id}")
    
    # Map environment names to the correct IDs
    if 'enhanced' in args.env.lower():
        args.env = 'EnhancedSafeHighway-v0'
        print(f"Using enhanced environment: {args.env}")
    elif 'highway' in args.env.lower() and 'v2' in args.env.lower():
        args.env = 'SafeHighwayV2-v0'
        print(f"Using V2 environment: {args.env}")
    elif 'highway' in args.env.lower():
        args.env = 'highway-safety-v0'
        print(f"Using standard environment: {args.env}")
    
    if args.policy == 'cvpo':
        if 'highway' in args.env.lower() or 'safe' in args.env.lower():
            config_path = osp.join(CONFIG_DIR, "config_highway.yaml")
        else:
            config_path = osp.join(CONFIG_DIR, "config_cvpo.yaml")
    else:
        config_path = osp.join(CONFIG_DIR, "config_baseline.yaml")
    
    config = load_config(config_path)
    config.update(args_dict)
    config["exp_name"] = gen_exp_name(config, args.suffix)
    config["data_dir"] = gen_data_dir_name(config)
    
    # Use SafetyGymRunner for all environments
    runner = SafetyGymRunner(**config)
    
    if args.mode == "train":
        runner.train()
    else:
        runner.eval(render=not args.no_render, sleep=args.sleep)
