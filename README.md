# cvpo-carla
A minimal framework for comparing CVPO with other safe reinforcement learning algorithms in the Carla driving simulator.

Make sure you have the Carla simulator installed
Install required dependencies:
Copypip install torch gym numpy matplotlib

Ensure all Safe RL libraries are installed

Start carla environment in a terminal:
cd carla
./CarlaUE4.sh -opengl -carla-server -RenderOffScreen

Train a Single Algorithm
export PYTHONPATH=$PYTHONPATH:/home/prakhargautam/experiments/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:/home/prakhargautam/experiments/cvpo-safe-rl/gym-carla

To train the CVPO algorithm (in a new terminal)
python run_carla_comparison.py --policy cvpo --epochs 300

Train and Compare Multiple Algorithms
To compare CVPO with SAC with Lagrangian constraints:
python compare_algorithms.py --algorithms cvpo sac_lag --epochs 300

Available Algorithms
cvpo: Debug CVPO implementation adapted for Carla
sac: Soft Actor-Critic (without safety constraints)
sac_lag: SAC with Lagrangian safety constraints
td3: Twin Delayed DDPG (without safety constraints)
td3_lag: TD3 with Lagrangian safety constraints

Command-line Options
Single Algorithm Training/Evaluation
python run_carla_comparison.py --policy POLICY [options]
Options:

--policy: Algorithm to use (default: cvpo)
--town: Carla town to use (default: Town05)
--mode: "train" or "eval" (default: train)
--epochs: Number of epochs to train (default: 300)
--cost_limit: Safety cost threshold (default: 50.0)
--seed: Random seed (default: 0)
--num_vehicles: Number of vehicles in environment (default: 0)
--desired_speed: Desired speed (default: 5.0)
--lane_threshold: Lane threshold (default: 3.0)
--device: Device to use ("cuda" or "cpu")
--verbose: Enable verbose output
--resume: Path to checkpoint to resume from
--output_dir: Directory to save outputs (default: carla_results)
--exp_name: Experiment name
--eval_episodes: Number of episodes to evaluate (default: 5)

Algorithm Comparison
python compare_algorithms.py --algorithms ALG1 ALG2 [options]
Options:

--algorithms: List of algorithms to compare (default: cvpo sac_lag)
--town: Carla town to use (default: Town05)
--epochs: Number of epochs to train (default: 300)
--seed: Random seed (default: 0)
--cost_limit: Safety cost threshold (default: 50.0)
--num_vehicles: Number of vehicles in environment (default: 0)
--desired_speed: Desired speed (default: 5.0)
--lane_threshold: Lane threshold (default: 3.0)
--output_dir: Output directory (default: carla_comparison)
--eval_after: Evaluate policies after training
--eval_episodes: Number of episodes for evaluation (default: 5)
--verbose: Enable verbose output
--device: Device to use ("cuda" or "cpu")

Example Workflows

Train CVPO with curriculum learning in Town05:
python run_carla_comparison.py --policy cvpo --town Town05 --epochs 300 --desired_speed 5.0

Compare CVPO with SAC-Lagrangian:
python compare_algorithms.py --algorithms cvpo sac_lag --epochs 300 --eval_after

Evaluate a trained model:
python run_carla_comparison.py --policy cvpo --mode eval --resume carla_results/cvpo_s0_Town05/model_epoch_300.pt


Output
Training and evaluation results will be saved in the specified output directory:

Model checkpoints: Saved as model_epoch_N.pt
Training logs: Including reward, cost, and other metrics
Videos: Recorded during training and evaluation (when enabled)

