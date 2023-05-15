##  TorchAudio’s basic I/O API to load audio files into PyTorch’s Tensor object, and save Tensor objects to audio files
import torch
import torchaudio
import io 
import boto3
import matplotlib.pyplot as plt

import requests
from botocore import UNSIGNED
from botocore.config import Config
from torchaudio.utils import download_asset
from IPython.display import Audio

### Downloading the audio files
SAMPLE_GSM = download_asset("tutorial-assets/steam-train-whistle-daniel_simon.gsm")
SAMPLE_WAV = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
SAMPLE_WAV_8000 = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042-8000hz.wav")

url = "https://download.pytorch.org/torchaudio/tutorial-assets/steam-train-whistle-daniel_simon.wav"
with requests.get(url, stream=True) as response:
    metadata = torchaudio.info(response.raw)
print(metadata)