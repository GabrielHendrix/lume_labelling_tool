## Installation

#### LLT needs to be installed first before use. The code requires cuda 12.1 [Installation](https://github.com/LumeRobotics/docs/blob/main/Installations/instaling_CUDA_12.1.md), `python>=3.10`, as well as `torch>=2.5.1` and `torchvision>=0.20.1`. 

First, install Python 3.10:

```bash
cd /opt
sudo wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tar.xz
sudo tar -xvf Python-3.10.0.tar.xz
cd Python-3.10.0
sudo ./configure --enable-optimizations
sudo make altinstall
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
sudo rm Python-3.10.0.tar.xz
```

Now it's time to install another important set of dependencies and the module:

```
cd lume_labellimg_tool
python3.10 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python3.10 -m pip install -e .
python3.10 -m pip install opencv-python
python3.10 -m pip install pyqt6==6.9.0
sudo apt install libxcb-cursor0
pip install -e .
cd checkpoints && ./download_ckpts.sh && cd ..

```
