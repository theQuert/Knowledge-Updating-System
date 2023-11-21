sudo apt update
sudo apt autoremove -y
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10 -y

# install cuda 12.3
# referece: https://developer.nvidia.com/cuda-downloads?target_os=Linux
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3

sudo apt-get update
sudo apt-get -y install cuda
sudo add-apt-repository -y ppa:graphics-drivers/ppa && sudo apt update
sudo apt -y install nvidia-cuda-toolkit
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
# check nvidia-smi and nvcc
nvidia-smi
nvcc -V

# prepare torch environment
sudo apt install -y python3-pip
sudo pip3 install torch torchvision torchaudio
# install venv
sudo apt install python3.10-venv