# cvpo-carla

A minimal framework for comparing CVPO with other safe reinforcement learning algorithms in the Carla driving simulator.

## Prerequisites

1.  Environment: Python 3.6 + conda. Use install_all.sh and requirements.txt in the root dir for installation. The following packages are installed for the purposes of this run.

```bash
# packages in environment at /home/prakhargautam/miniconda3/envs/carla_venv_py36:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
absl-py                   1.4.0                    pypi_0    pypi
aiosignal                 1.2.0                    pypi_0    pypi
alsa-lib                  1.2.3.2              h166bdaf_0    conda-forge
attrs                     22.2.0                   pypi_0    pypi
bzip2                     1.0.8                h7f98852_4    conda-forge
c-ares                    1.18.1               h7f98852_0    conda-forge
ca-certificates           2025.2.25            h06a4308_0  
cachetools                4.2.4                    pypi_0    pypi
cairo                     1.16.0            h6cf1ce9_1008    conda-forge
certifi                   2021.5.30        py36h06a4308_0  
charset-normalizer        2.0.12                   pypi_0    pypi
click                     8.0.4                    pypi_0    pypi
cpprb                     10.1.1                   pypi_0    pypi
cycler                    0.11.0                   pypi_0    pypi
dataclasses               0.8                      pypi_0    pypi
dbus                      1.13.6               h48d8840_2    conda-forge
decorator                 4.4.2                    pypi_0    pypi
distlib                   0.3.9                    pypi_0    pypi
easydict                  1.9                      pypi_0    pypi
evdev                     1.6.1                    pypi_0    pypi
expat                     2.4.8                h27087fc_0    conda-forge
ffmpeg                    4.3.2                hca11adc_0    conda-forge
filelock                  3.4.1                    pypi_0    pypi
fontconfig                2.14.0               h8e229c2_0    conda-forge
freetype                  2.10.4               h0708190_1    conda-forge
frozenlist                1.2.0                    pypi_0    pypi
gettext                   0.19.8.1          h0b5b191_1005    conda-forge
glib                      2.68.4               h9c3ff4c_0    conda-forge
glib-tools                2.68.4               h9c3ff4c_0    conda-forge
gmp                       6.2.1                h58526e2_0    conda-forge
gnutls                    3.6.13               h85f3911_1    conda-forge
google-auth               2.22.0                   pypi_0    pypi
google-auth-oauthlib      0.4.6                    pypi_0    pypi
graphite2                 1.3.13            h58526e2_1001    conda-forge
grpcio                    1.48.2                   pypi_0    pypi
gst-plugins-base          1.18.5               hf529b03_0    conda-forge
gstreamer                 1.18.5               h76c114f_0    conda-forge
gym                       0.12.5                   pypi_0    pypi
harfbuzz                  2.9.1                h83ec7ef_1    conda-forge
hdf5                      1.12.1          nompi_h2750804_100    conda-forge
icu                       68.2                 h9c3ff4c_0    conda-forge
idna                      3.10                     pypi_0    pypi
imageio                   2.15.0                   pypi_0    pypi
importlib-metadata        4.8.3                    pypi_0    pypi
importlib-resources       5.4.0                    pypi_0    pypi
jasper                    1.900.1           h07fcdf6_1006    conda-forge
jbig                      2.1               h7f98852_2003    conda-forge
joblib                    1.1.1                    pypi_0    pypi
jpeg                      9e                   h166bdaf_1    conda-forge
jsonschema                3.2.0                    pypi_0    pypi
keyutils                  1.6.1                h166bdaf_0    conda-forge
kiwisolver                1.3.1                    pypi_0    pypi
krb5                      1.19.3               h3790be6_0    conda-forge
lame                      3.100             h7f98852_1001    conda-forge
ld_impl_linux-64          2.40                 h12ee557_0  
lerc                      2.2.1                h9c3ff4c_0    conda-forge
libblas                   3.9.0           16_linux64_openblas    conda-forge
libcblas                  3.9.0           16_linux64_openblas    conda-forge
libclang                  11.1.0          default_ha53f305_1    conda-forge
libcurl                   7.79.1               h2574ce0_1    conda-forge
libdeflate                1.7                  h7f98852_5    conda-forge
libedit                   3.1.20191231         he28a2e2_2    conda-forge
libev                     4.33                 h516909a_1    conda-forge
libevent                  2.1.10               h9b69904_4    conda-forge
libffi                    3.3                  he6710b0_2  
libgcc-ng                 11.2.0               h1234567_1  
libgfortran-ng            13.2.0               h69a702a_0    conda-forge
libgfortran5              13.2.0               ha4646dd_0    conda-forge
libglib                   2.68.4               h3e27bee_0    conda-forge
libgomp                   11.2.0               h1234567_1  
libiconv                  1.17                 h166bdaf_0    conda-forge
liblapack                 3.9.0           16_linux64_openblas    conda-forge
liblapacke                3.9.0           16_linux64_openblas    conda-forge
libllvm11                 11.1.0               hf817b99_2    conda-forge
libnghttp2                1.43.0               h812cca2_1    conda-forge
libogg                    1.3.4                h7f98852_1    conda-forge
libopenblas               0.3.21               h043d6bf_0  
libopencv                 4.5.3            py36h9976982_2    conda-forge
libopus                   1.3.1                h7f98852_1    conda-forge
libpng                    1.6.37               h21135ba_2    conda-forge
libpq                     13.3                 hd57d9b9_0    conda-forge
libprotobuf               3.16.0               h780b84a_0    conda-forge
libssh2                   1.10.0               ha56f1ee_2    conda-forge
libstdcxx-ng              11.2.0               h1234567_1  
libtiff                   4.3.0                hf544144_1    conda-forge
libuuid                   2.32.1            h7f98852_1000    conda-forge
libvorbis                 1.3.7                h9c3ff4c_0    conda-forge
libwebp-base              1.2.2                h7f98852_1    conda-forge
libxcb                    1.13              h7f98852_1004    conda-forge
libxkbcommon              1.0.3                he3ba5ed_0    conda-forge
libxml2                   2.9.12               h72842e0_0    conda-forge
lz4-c                     1.9.3                h9c3ff4c_1    conda-forge
markdown                  3.3.7                    pypi_0    pypi
matplotlib                3.3.4                    pypi_0    pypi
msgpack                   1.0.5                    pypi_0    pypi
mysql-common              8.0.25               ha770c72_2    conda-forge
mysql-libs                8.0.25               hfa10184_2    conda-forge
ncurses                   6.4                  h6a678d5_0  
nettle                    3.6                  he412f7d_0    conda-forge
networkx                  2.5.1                    pypi_0    pypi
nspr                      4.32                 h9c3ff4c_1    conda-forge
nss                       3.69                 hb5efdd6_1    conda-forge
numpy                     1.19.5                   pypi_0    pypi
oauthlib                  3.2.2                    pypi_0    pypi
opencv                    4.5.3            py36h5fab9bb_2    conda-forge
openh264                  2.1.1                h780b84a_0    conda-forge
openssl                   1.1.1w               h7f8727e_0  
packaging                 21.3                     pypi_0    pypi
pandas                    1.1.5                    pypi_0    pypi
pcre                      8.45                 h9c3ff4c_0    conda-forge
pillow                    8.4.0                    pypi_0    pypi
pip                       21.2.2           py36h06a4308_0  
pixman                    0.40.0               h36c2ea0_0    conda-forge
platformdirs              2.4.0                    pypi_0    pypi
protobuf                  3.19.6                   pypi_0    pypi
pthread-stubs             0.4               h36c2ea0_1001    conda-forge
py-opencv                 4.5.3            py36he75451f_2    conda-forge
pyasn1                    0.5.1                    pypi_0    pypi
pyasn1-modules            0.3.0                    pypi_0    pypi
pygame                    1.9.6                    pypi_0    pypi
pyglet                    2.0.10                   pypi_0    pypi
pynput                    1.7.3                    pypi_0    pypi
pyparsing                 3.1.4                    pypi_0    pypi
pyrsistent                0.18.0                   pypi_0    pypi
python                    3.6.13               h12debd9_1  
python-dateutil           2.9.0.post0              pypi_0    pypi
python-xlib               0.33                     pypi_0    pypi
python_abi                3.6                     2_cp36m    conda-forge
pytz                      2025.1                   pypi_0    pypi
pywavelets                1.1.1                    pypi_0    pypi
pyyaml                    5.4.1                    pypi_0    pypi
qt                        5.12.9               hda022c4_4    conda-forge
ray                       2.4.0                    pypi_0    pypi
readline                  8.2                  h5eee18b_0  
requests                  2.27.1                   pypi_0    pypi
requests-oauthlib         2.0.0                    pypi_0    pypi
rsa                       4.9                      pypi_0    pypi
scikit-image              0.16.2                   pypi_0    pypi
scipy                     1.5.4                    pypi_0    pypi
seaborn                   0.11.2                   pypi_0    pypi
setuptools                58.0.4           py36h06a4308_0  
six                       1.17.0                   pypi_0    pypi
sqlite                    3.45.3               h5eee18b_0  
tabulate                  0.8.9                    pypi_0    pypi
tensorboard               2.10.1                   pypi_0    pypi
tensorboard-data-server   0.6.1                    pypi_0    pypi
tensorboard-plugin-wit    1.8.1                    pypi_0    pypi
tensorboardx              2.6                      pypi_0    pypi
tk                        8.6.14               h39e8969_0  
torch                     1.10.2                   pypi_0    pypi
tqdm                      4.62.3                   pypi_0    pypi
typing-extensions         4.1.1                    pypi_0    pypi
urllib3                   1.26.20                  pypi_0    pypi
virtualenv                20.17.1                  pypi_0    pypi
werkzeug                  2.0.3                    pypi_0    pypi
wheel                     0.37.1             pyhd3eb1b0_0  
x264                      1!161.3030           h7f98852_1    conda-forge
xorg-kbproto              1.0.7             h7f98852_1002    conda-forge
xorg-libice               1.0.10               h7f98852_0    conda-forge
xorg-libsm                1.2.3             hd9c2040_1000    conda-forge
xorg-libx11               1.7.2                h7f98852_0    conda-forge
xorg-libxau               1.0.9                h7f98852_0    conda-forge
xorg-libxdmcp             1.1.3                h7f98852_0    conda-forge
xorg-libxext              1.3.4                h7f98852_1    conda-forge
xorg-libxrender           0.9.10            h7f98852_1003    conda-forge
xorg-renderproto          0.11.1            h7f98852_1002    conda-forge
xorg-xextproto            7.3.0             h7f98852_1002    conda-forge
xorg-xproto               7.0.31            h7f98852_1007    conda-forge
xz                        5.6.4                h5eee18b_1  
zipp                      3.6.0                    pypi_0    pypi
zlib                      1.2.13               h5eee18b_1  
zstd                      1.5.0                ha95c52a_0    conda-forge
```
3.  GPU installed with CUDA (verify with nvidia-smi for installation)
4.  **Carla Simulator:** Ensure you have the Carla simulator installed. Refer to the official Carla documentation for installation instructions. This example uses CARLA_0.9.15.tar.gz.
5.  **Dependencies:** Additionally install the following packages:

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
