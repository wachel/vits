import os
import sys
import torch
import torch.nn.functional as F
import librosa

import numpy as np

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    Wav2Vec2Model,
)
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

model_path="TencentGameMate/chinese-wav2vec2-base"
device = 'cuda'

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
model = Wav2Vec2Model.from_pretrained(model_path,)

# for pretrain: Wav2Vec2ForPreTraining
# model = Wav2Vec2ForPreTraining.from_pretrained(model_path)

model = model.to(device)
model = model.half()
model.eval()

#从wav中提取语义数据，并保存到.semantic.npy中
def extract_semantic_features(wav_path:str):
    np_path = os.path.splitext(wav_path)[0] + '.semantic.pt'
    if os.path.exists(np_path):
        return
    print(wav_path)

    wav, sr = librosa.load(wav_path, sr=16000)
    
    input_values = feature_extractor(wav, return_tensors="pt",sampling_rate=sr).input_values
    input_values = input_values.half()
    input_values = input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        #print(outputs.keys())
        #last_hidden_state = outputs.last_hidden_state
        extract_features = outputs.extract_features.squeeze(0)
        #np.savez_compressed(np_path, extract_features)
        torch.save(extract_features, np_path)

#提取wav2vec2的last_hidden_state，用来做语气分析等，一句话对应一个
def extract_wav2vec2_hidden_state(wav_path:str):
    np_path = os.path.splitext(wav_path)[0] + '.w2v_hm.pt'
    if os.path.exists(np_path):
        return
    print(wav_path)

    wav, sr = librosa.load(wav_path, sr=16000)
    input_values = feature_extractor(wav, return_tensors="pt",sampling_rate=sr).input_values
    input_values = input_values.half()
    input_values = input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        #print(outputs.keys())
        last_hidden_state = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        #extract_features = outputs.extract_features.squeeze(0)
        torch.save(last_hidden_state, np_path)

def extract_all(root_path:str):
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                extract_wav2vec2_hidden_state(wav_path)
    
#wav_path="/data/home/hailongwang/gits/tts_test/data/qs/wav_16k/BaiFeng/10_DongHaiZhiBin_DHZB_XZX_216.wav"
#extract_semantic_features(wav_path)
#extract_all('data/qs/wav_16k')
#extract_all('data/cp2077/wav_16k')

if __name__ == '__main__':
    # if(len(sys.argv)>1):
    #     path = sys.argv[1]
    #     extract_all(path)
    # else:
    extract_all('data/cp2077/wav_22k')
    extract_all('data/qs/wav_22k')