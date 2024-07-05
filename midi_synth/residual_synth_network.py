# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch nn network for predicting synth params for residual guitar sounds from overall MIDI conditioning

import torch.nn as nn
import math
from globals import *

class ResidualSynthNetwork(nn.Module):
    """
    Torch nn decoder for taking in overall guitar midi conditioning 
    and outputting synth params for residual guitar sounds.
    """
    def __init__(self,
                linear_1_size=MONO_NET_SM_LINEAR_1_SIZE,
                linear_2_size=MONO_NET_SM_LINEAR_2_SIZE,
                gru_num_units=MONO_NET_SM_GRU_NUM_UNITS,
                num_oscillators=NUM_OSCILLATORS,
                num_filter_coeffs=NUM_FILTER_COEFFICIENTS,
                use_mfcc_input=MIDI_SYNTH_USE_MFCC_INPUT):
        """
        Initialize Residual Synth Network module.


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

        # create output linear layers
        self.dense_amplitudes = nn.Linear(linear_2_size, num_oscillators + 1) # plus 1 for overall amplitude
        self.dense_filter_coeffs = nn.Linear(linear_2_size, num_filter_coeffs)
        self.dense_f0 = nn.Linear(linear_2_size, 1)

        # modified sigmoid output activiation
        # as in DDSP paper
        self.modified_sigmoid = lambda x: 2 * torch.sigmoid(x) ** (math.log(10)) + 1e-7 # [0,2]

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
        amplitude_tensor =  self.modified_sigmoid(self.dense_amplitudes(layer_norm_out))
        H = self.modified_sigmoid(self.dense_filter_coeffs(layer_norm_out))
        f0 = torch.sigmoid(self.dense_f0(layer_norm_out).flatten(1,2)) * (SR/2) # sigmoid * nyquist to get full range of freqs, flatten to remove final dim of 1 and get shape: (batch, frames) 
        
        # treat first amplitude as overall, rest as harmonic distribution
        # force harm_distr to sum to one
        overall_amplitude = amplitude_tensor[..., 0]
        harm_distr =  amplitude_tensor[..., 1:] # removed softmax (normalization now in harm osc, post antialiasing)
        harm_distr = torch.permute(harm_distr, (0,2,1))  # make it be (batch, num_oscillators, frames)

        # create output dict
        p = {}
        
        # store results in dict and return
        p['overall_amplitude'] = overall_amplitude
        p['harm_distr'] = harm_distr
        p['H'] = H
        p['f0'] = f0


        return p