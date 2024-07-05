# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch nn network for predicting stft of residual guitar sounds from overall MIDI conditioning

import torch.nn as nn
import torchaudio
import math
from globals import *
from util import numpy_to_torch

class ResidualSynthUnconstrained(nn.Module):
    """
    Torch nn decoder for taking in overall guitar midi conditioning 
    and outputting predicted audio of residual guitar sounds, via stft.
    """
    def __init__(self, 
                linear_1_size=RESIDUAL_LINEAR_1_SIZE,
                linear_2_size=RESIDUAL_LINEAR_2_SIZE,
                gru_num_units=RESIUDAL_GRU_NUM_UNITS,
                use_mfcc_input=MIDI_SYNTH_USE_MFCC_INPUT):
        """
        Initialize Residual Synth Unconstrained module.

        Parameters
        ----------
        linear_1_size : int
            size of 1st dense layer
        linear_2_size : int
            size of 2nd dense layer
        gru_num_units : int
            unit size of gru
        num_oscillators : int
            number of oscillators to get harm distribution for (corresponds to number of oscillators in synth)
        num_filter_coeffs : int
            number of filter coefficients to output for use in filtered noise synth
        use_mfcc_input : bool
            if true, then synth expects audio as input so that mfccs can be calculated
        """
        super().__init__()

        # set input/output feature size
        self.num_inputs = 12 # for pitch and velocity for each string

        # optionally include mfcc as an input
        self.use_mfcc_input = use_mfcc_input
        if self.use_mfcc_input:
            self.num_inputs += N_MFCC

        # create layers
        self.linear1 = nn.Linear(self.num_inputs, linear_1_size)
        self.leakyReLU1 = nn.LeakyReLU()
        self.gru = nn.GRU(input_size=linear_1_size,
                        hidden_size=gru_num_units,
                        batch_first=True)
        self.linear2 = nn.Linear(gru_num_units, linear_2_size)
        self.leakyReLU2 = nn.LeakyReLU()
        self.layer_norm = nn.LayerNorm(linear_2_size)

        # create output linear layer that predicts the power spectrogram frames
        # if stft is taken with n_fft=2048 for instance,
        # the result has 1025 freq bins. So n_fft//2 + 1 is how many values to predict.
        self.dense_stft = nn.Linear(linear_2_size, (N_FFT//2 + 1))

        # inverse spectrogram transform
        self.griffinlim = torchaudio.transforms.GriffinLim(
            n_fft = N_FFT,
            hop_length = HOP_LENGTH,
            power = 2.0, 
        )

    def forward(self, conditioning, mfcc=None):
        """
        Process string conditioning and idx outputs synth params.

        Parameters
        ----------
        conditioning : tensor (batch, frames, num_strings=6, pitch/vel=2)
            pitch and onset velocity labels across 6 strings
            assumed to be normalized in [0,1]
        mfcc : tensor (batch, frames, n_mfcc)
            batch of mfccs for the input audio

        Returns
        ----------
            p['H'] : tensor (batch, frames, filter_len)
                output noise filter coefficients
            p['overall_amplitude'] : tensor (batch, frames)
                overall amplitude values
            p['harm_distr'] : tensor (batch, # harmonics, frames)
                harmonic distribution (sum to 1) envelope tensor
            p['f0'] : tensor (batch, frames)
                f0, predicted
            
        """
        # set up inputs
        inputs = conditioning.flatten(2,3) # shape: (batch, frames, 12)

        # if using mfcc, concatenate onto inputs
        if self.use_mfcc_input:
            inputs = torch.cat([inputs, mfcc], -1)

        # feed inputs thru
        linear1_out = self.linear1(inputs)
        leaky1_out = self.leakyReLU1(linear1_out)
        gru_out = self.gru(leaky1_out)[0]
        linear2_out = self.linear2(gru_out)
        leaky2_out = self.leakyReLU2(linear2_out)
        layer_norm_out = self.layer_norm(leaky2_out)

        # get amplitude tensor, filter coeffs H, and midi_pitch with final dense layers + modified sigmoid
        stft_frames = torch.sigmoid(self.dense_stft(layer_norm_out)) # (batch, frames, n_fft)
        
        # permute to get dimensions right for griffin lim 
        spec = torch.permute(stft_frames, (0,2,1))  # make it be (batch, n_fft, frames)

        # invert spectrogram via griffin lim
        audio = self.griffinlim(spec)

        # ## TESTING, temporarily removed griffin lim
        # audio = numpy_to_torch(np.zeros((conditioning.shape[0], int(SR * ITEM_DUR))))

        return audio