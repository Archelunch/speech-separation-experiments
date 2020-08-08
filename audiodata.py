import pandas as pd
import random

import torch

from torch.utils.data import Dataset
import torchaudio


class SpeechDataset(Dataset):
    """Wrapper for speech datasets"""

    def __init__(self, root_dir: str, meta_info: str, transform: torch.nn.Sequential = None, speakers_count: int = 2, sample_rate: int = 16000, duration: int = 4):
        self.root_dir = root_dir
        self.meta_info = pd.read_csv(f"{root_dir}/{meta_info}", sep='\t')
        self.transform = transform
        self.speakers_count = speakers_count
        self.sample_rate = sample_rate
        self.duration = duration
        self.speaker_dict = {us: e for e, us in enumerate(
            list(self.meta_info.client_id.unique()))}

    def __len__(self):
        return int(len(self.meta_info)/self.speakers_count)

    def get_audio(self, audio_name: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(
            f'{self.root_dir}/clips/{audio_name}')
        if sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(
                sample_rate, self.sample_rate)(waveform)
        return self.transform(waveform) if self.transform else waveform

    def crop_audio(self, audio: torch.Tensor) -> torch.Tensor:
        temp = torch.zeros((1, self.sample_rate*self.duration))
        if audio.size(1) < self.sample_rate*self.duration:
            offset = random.randint(0, temp.size(1)-audio.size(1))
            temp[0, offset:audio.size(1)+offset] = audio[0]
        else:
            offset = random.randint(0, audio.size(1) - temp.size(1))
            temp[0] = audio[0, offset:temp.size(1)+offset]
        return temp

    def mix_audio(self, audios: torch.Tensor) -> torch.Tensor:
        return audios.sum(dim=0)/self.speakers_count

    def __getitem__(self, idx):
        samples = self.meta_info.sample(self.speakers_count)
        speakers = torch.LongTensor([self.speaker_dict[s]
                                     for s in samples['client_id'].values])
        audios = torch.stack([self.crop_audio(self.get_audio(audio))
                              for audio in samples['path'].values])
        mixed_audio = self.mix_audio(audios)
        return {'speakers': speakers, 'audio_input': mixed_audio, 'audio_targets': audios}
