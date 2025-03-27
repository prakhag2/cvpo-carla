# cvpo-carla

A minimal framework for comparing CVPO with other safe reinforcement learning algorithms in the Carla driving simulator.

## Prerequisites

1.  Environment: Python 3.6 + conda. Use install_all.sh and requirements.txt in the root dir for installation.
2.  GPU installed with CUDA (verify with nvidia-smi for installation)
3.  **Carla Simulator:** Ensure you have the Carla simulator installed. Refer to the official Carla documentation for installation instructions. This example uses CARLA_0.9.15.tar.gz.
4.  **Dependencies:** Additionally install the following packages:

    ```bash
    pip install torch gym numpy matplotlib
    ```

## Setup

1.  **Start Carla Server:** Open a terminal and start the Carla server in headless mode:

    ```bash
    cd carla
    ./CarlaUE4.sh -opengl -carla-server -RenderOffScreen
    ```

2.  **Environment Variables:** On a different terminal set the `PYTHONPATH` to include the necessary Carla and gym-carla directories. Adjust the paths according to your installation:

    ```bash
    cd cvpo-safe-rl
    export PYTHONPATH=$PYTHONPATH:/home/prakhargautam/experiments/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:/home/prakhargautam/experiments/cvpo-safe-rl/gym-carla
    ```

## Training and Evaluation

### Train a Single Algorithm

1.  **Run Training:** Open a new terminal and execute the `run_carla_comparison.py` script. For example, to train the CVPO algorithm:

    ```bash
    python run_carla_comparison.py --policy cvpo --epochs 300
    ```

### Train and Compare Multiple Algorithms

1.  **Run Comparison:** To compare multiple algorithms, use the `compare_algorithms.py` script. For example, to compare CVPO with SAC with Lagrangian constraints:

    ```bash
    python compare_algorithms.py --algorithms cvpo sac_lag --epochs 300
    ```

## Available Algorithms

* `cvpo`: Debug CVPO implementation adapted for Carla.
* `sac`: Soft Actor-Critic (without safety constraints).
* `sac_lag`: SAC with Lagrangian safety constraints.
* `td3`: Twin Delayed DDPG (without safety constraints).
* `td3_lag`: TD3 with Lagrangian safety constraints.

## Command-line Options

### Single Algorithm Training/Evaluation

```bash
python run_carla_comparison.py --policy POLICY [options]
