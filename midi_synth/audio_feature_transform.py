# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch synth module with midi-like conditioning, predicting string params in unison

import torch.nn as nn
import torchaudio.transforms
import save_load
from util import crop_or_pad
from globals import *

class AudioFeatureTransform(nn.Module):
    """
    Torch nn module for transforming audio into an audio feature.
    """
    def __init__(self,
                mode=MIDI_SYNTH_POLY_INPUT_AUDIO_FEATURE,
                sr=SR,
                hop_length=HOP_LENGTH,
                n_fft=N_FFT,
                n_mels=N_MELS,
                n_mfcc=N_MFCC):
        """
        Initialize AudioFeatureTransform module.

        Parameters
        ----------
        mode : str
            feature to extract: "stft" (spectrogram), "melspec", "mfcc"
        sr : float
            audio sampling rate
        hop_lenth : int
            samples to hop between frame starts
        n_fft : int
            size of fft
        n_mels : int
            number of mel bins
        n_mfcc : int
            number of mel frequency cepstral coefficients   
        """
        super().__init__()

        self.mode = mode

        if self.mode == "stft":
            self.transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, 
                                                               hop_length=hop_length)
            self.feature_size = (n_fft // 2) + 1

        elif self.mode == "melspec":
            self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr, 
                                                                n_fft=n_fft, 
                                                                hop_length=hop_length, 
                                                                n_mels=n_mels)
            self.feature_size = n_mels

        elif self.mode == "mfcc":
            self.transform = torchaudio.transforms.MFCC(sample_rate=sr,
                                                        n_mfcc=n_mfcc,
                                                        melkwargs={"n_fft": n_fft, 
                                                            "hop_length": hop_length, 
                                                            "n_mels": n_mels}
            )
            self.feature_size = n_mfcc

        else:
            self.feature_size = 0

    def forward(self, audio, num_frames=None):
        """
        Process a batch of audio to get features.

        Parameters
        ----------
        audio : tensor (batch, samples)
            tensor of audio to extract features from
        num_frames : int
            desired number of frames for audio feature (to crop to)

        Returns
        ----------
        features : tensor (batch, frames, features)
        """

        features = self.transform(audio) # (batch, features, frames)

        if num_frames is not None:
            features = crop_or_pad(features, num_frames)

        # make features dimension last
        features = torch.transpose(features, 1, 2) # (batch, frames, features)

        return features
    


