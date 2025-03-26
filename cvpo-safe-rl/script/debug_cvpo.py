# debug_cvpo.py
import numpy as np
from safe_rl.policy.cvpo import CVPO

# Override the act method to include debugging
class DebugCVPO(CVPO):
    def act(self, obs, deterministic=False, with_logprob=False):
        '''
        Debugging version of the act method
        '''
        print(f"CVPO.act received obs: {obs}")
        print(f"CVPO.act obs type: {type(obs)}")
        
        # Handle the case where obs is a tuple (new Gym API)
        if isinstance(obs, tuple):
            print("Extracting observation from tuple")
            # Extract just the observation part
            obs = obs[0]
            
        if isinstance(obs, np.ndarray):
            print(f"CVPO.act obs shape: {obs.shape}")
        
        # Ensure obs is a proper numpy array before proceeding
        if not isinstance(obs, np.ndarray):
            print("Converting obs to numpy array")
            obs = np.array(obs, dtype=np.float32)
        
        # Ensure obs has the right shape
        if len(obs.shape) == 1:
            print("Reshaping 1D array to 2D")
            obs = np.reshape(obs, (1, -1))  # Add batch dimension
        
        # Now call the parent method using the properly formatted obs
        try:
            return super().act(obs, deterministic, with_logprob)
        except Exception as e:
            print(f"Error in CVPO.act: {e}")
            print(f"Obs after formatting: {obs}")
            print(f"Obs shape after formatting: {obs.shape}")
            raise
