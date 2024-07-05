# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch nn network for decoding 6 string conditioning (+ audio features) into synth params

import torch.nn as nn
import torch.nn.functional as F
from midi_synth.midi_util import midi_to_hz
import math
from networks.basic_gru import BasicGRU
from networks.basic_transformer import BasicTransformer
from networks.transformer_2D import Transformer2D
from globals import *

class PolyphonicNet(nn.Module):
    """
    Torch nn decoder for taking full guitar midi conditioning (and optionally audio features) and predicting all string synth params.
    """
    def __init__(self,
                model_version=POLY_NET_MODEL_VERSION,
                num_oscillators=NUM_OSCILLATORS,
                num_filter_coeffs=NUM_FILTER_COEFFICIENTS,
                audio_feature_size=0):
        """
        Initialize DDSP decoder module.


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
        audio_feature_size : int
            size of input audio feature used
        """
        super().__init__()

        # set input/output feature size
        self.num_inputs = 12 + audio_feature_size # 12 for pitch and velocity for 6 strings
        self.num_amplitude_outputs = (num_oscillators + 1) * 6 # plus 1 for overall amplitude, x6 for 6 strings
        self.num_filter_outputs = (num_filter_coeffs) * 6 # plus 1 for overall amplitude, x6 for 6 strings

        # create model
        if model_version == "basic-gru":
            self.nn_model = BasicGRU(self.num_inputs)
        elif model_version == "basic-transformer":
            self.nn_model = BasicTransformer(self.num_inputs)
        elif model_version == "transformer-2D":
            self.nn_model = Transformer2D(self.num_inputs)

        # get num outputs from chosen model
        self.num_outputs = self.nn_model.num_outputs

        # create output linear layers
        self.dense_amplitudes = nn.Linear(self.num_outputs, self.num_amplitude_outputs) # plus 1 for overall amplitude
        self.dense_filter_coeffs = nn.Linear(self.num_outputs, self.num_filter_outputs)

        # modified sigmoid output activiation
        # as in DDSP paper
        self.modified_sigmoid = lambda x: 2 * torch.sigmoid(x) ** (math.log(10)) + 1e-7


    def forward(self, conditioning, audio_feature):
        """
        Process string conditioning and idx outputs synth params.

        Parameters
        ----------
        conditioning : tensor (batch, frames, num_string=6, pitch/vel=2)
            pitch and onset velocity labels for all guitar strings
            assumed to be normalized in [0,1]
        audio_feature : tensor (batch, frames, feature_size)
            input audio features to use as additional input

            
        Returns
        ----------
            p['H'] : tensor (batch x 6, frames, filter_len)
                output noise filter coefficients
            p['overall_amplitude'] : tensor (batch x 6, frames)
                overall amplitude values
            p['harm_distr'] : tensor (batch x 6, # harmonics, frames)
                harmonic distribution envelope tensor
            p['f0'] : tensor (batch x 6, frames)
                f0, from input conditioning    
        """
        # flatten last two dims
        inputs = conditioning.flatten(2,3) # (batch, frames, 12)
        
        # if using audio features, concatenate onto inputs
        if audio_feature is not None:
            inputs = torch.cat([inputs, audio_feature], -1)
            
        # feed inputs thru model
        model_outputs = self.nn_model(inputs)

        # get amplitude tensor and filter coeffs H with final dense layers + modified sigmoid
        amplitude_tensor =  self.modified_sigmoid(self.dense_amplitudes(model_outputs)) # (batch, frames, num_amplitude_outputs)
        H = self.modified_sigmoid(self.dense_filter_coeffs(model_outputs)) # (batch, frames, num_filter_outputs)
        
        # reshape amplitudes and H to be (6 x batch, frames, string_outputs)
        amplitude_tensor = amplitude_tensor.unflatten(2, (6, -1)) # (batch, frames, 6, amplitudes)
        amplitude_tensor = amplitude_tensor.permute(2, 0, 1, 3).flatten(0, 1) # (batch x 6, frames, ampltidues)
        H = H.unflatten(2, (6, -1)) # (batch, frames, 6, filter_ceoffs)
        H = H.permute(2, 0, 1, 3).flatten(0, 1) # (batch x 6, frames, filter_coeffs)

        # treat first amplitude as overall, rest as harmonic distribution
        # force harm_distr to sum to one
        overall_amplitude = amplitude_tensor[..., 0] # (batch x 6, frames)
        harm_distr =  amplitude_tensor[..., 1:] # (batch x 6, frames, num_oscillators)
        harm_distr = torch.permute(harm_distr, (0,2,1))  # make it be (batch x 6, num_oscillators, frames)

        # add f0 (converted from midi pitch) to params dict so that it can be used by the oscillator
        midi_pitch = conditioning[:,:,:,0] * MIDI_NORM # scale from [0,1] to [0,127], (batch, frames, string=6)
        f0 = midi_to_hz(midi_pitch)
        f0[torch.where(midi_pitch <= 0.0)] = 0 # midi pitch values less than or equal 0.0 should become 0 Hz
        f0 = f0.permute(2, 0, 1).flatten(0, 1) # (batch x 6, frames)

        # create a string mask from the midi pitch conditioning
        # use this to zero out all param values when the midi conditioning is zero
        if MASK_STRING_DATA_WITH_CONDITIONING:
            # generate mask using f0 (which is zero/non zero tracking string activity)
            string_mask = torch.zeros_like(f0) # (batch x 6, frames), values [0,1]
            string_mask[torch.where(f0 > 0.0)] = 1.0 # set one for all locations where f0 is nonzero

            # widen nonzero areas of string mask
            # sum together shifted copies and clip to [0,1]
            if EXTEND_STRING_MASK:
                left_extend = STRING_MASK_LEFT_EXTEND_FRAMES
                right_extend = STRING_MASK_RIGHT_EXTEND_FRAMES
                string_mask_left_shift = F.pad(string_mask[:,left_extend:], (0,left_extend))
                if right_extend > 0: # only try to right shift if nonzero
                    string_mask_right_shift = F.pad(string_mask[:,:(right_extend * -1)], (right_extend,0))
                else: # if no right shift, don't need to slice/pad
                    string_mask_right_shift = string_mask
                string_mask_sum = string_mask_left_shift + string_mask_right_shift
                string_mask = torch.clip(string_mask_sum, 0, 1)


            # apply mask to all other params
            overall_amplitude = overall_amplitude * string_mask # (batch, frames)
            H = H * string_mask[:, :, None] # (batch, frames, filter_len)

        # create output dict
        p = {}
        
        # store results in dict and return
        p['overall_amplitude'] = overall_amplitude  # (batch x 6, frames)
        p['harm_distr'] = harm_distr                # (batch x 6, num_harmonics, frames)
        p['H'] = H                                  # (batch x 6, frames, filter_len)
        p['f0'] = f0                                # (batch x 6, frames)

        return p