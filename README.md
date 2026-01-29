## Setup

```bash
conda create -n pytorch_gpu python=3.12 -y
conda activate pytorch_gpu
```
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install matplotlib numpy pillow opencv-python tqdm ipykernel pandas
```
