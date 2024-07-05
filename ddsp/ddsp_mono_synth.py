# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch ddsp mono synth module includes decoder and harmonic + noise synth

from ddsp.ddsp_decoder import DDSPDecoder
from ddsp.harmonic_oscillator import HarmonicOscillator
from ddsp.filtered_noise import FilteredNoise
from ddsp.trainable_reverb import TrainableReverb
from ddsp.multi_scale_spectral_loss import multi_scale_spectral_loss
from globals import *


class DDSPMonoSynth(nn.Module):
    """
    Torch nn module incorporating the ddsp decoder and the harmonic + noise synthesizers.
    """
    def __init__(self,
                sr=SR, 
                hop_length=HOP_LENGTH,
                use_timbre=False, 
                mlp_num_units=MLP_NUM_UNITS, 
                mlp_num_layers=MLP_NUM_LAYERS, 
                mlp_activation=MLP_ACTIVATION,
                gru_num_units=GRU_NUM_UNITS,
                num_oscillators=NUM_OSCILLATORS,
                num_filter_coeffs=NUM_FILTER_COEFFICIENTS,
                use_reverb=DDSP_MONO_SYNTH_USE_REVERB,
                eval_dict=DDSP_MONO_SYNTH_EVAL_DICT):
        """
        Initialize DDSP mono synth module.


        Parameters
        ----------
        sr : float
            audio sampling rate
        hop_lenth : int
            samples to hop between frame starts
        use_timbre : bool
		    whether or not timbre is provided as an input
        mlp_num_units : int
            number of units in each mlp layer
        mlp_num_layers : int
            number of layers in each mlp
        mlp_activation : torch.nn module
            activation to apply in mlp layers
        gru_num_units : int
            number of units in the gru's hidden layer
        num_oscillators : int
            number of oscillators to get harm distribution for (corresponds to number of oscillators in synth)
        filter_coeffs : int
            number of filter coefficients to output for use in filtered noise synth
        eval_dict : dictionary
            contains flags for which evaluations to carry out
        """
        super().__init__()

        

        ####### All models should have the following defined ########

        self.name = "DDSP Mono Synth"

        # define keys to access inputs, outputs + labels (targets) of the network
        self.config = {'inputs' : ["f0", "loudness"], # takes in f0 and loudness (or onset_vel)
                        'labels' : ["audio"], # compare against ground truth audio
                        'outputs' : ["audio"], # model outputs synthesized audio, will be overwritten below if reverb is used               
        }

        # dictionary of evaluations to include in checkpoint
        self.eval_dict = eval_dict

        ##############################################################

        self.use_reverb = use_reverb

        self.loss_function = multi_scale_spectral_loss

        # create decoder
        self.decoder = DDSPDecoder(use_timbre=False,
                                mlp_num_units=mlp_num_units, 
                                mlp_num_layers=mlp_num_layers, 
                                mlp_activation= mlp_activation,
                                gru_num_units=gru_num_units,
                                num_oscillators=num_oscillators,
                                num_filter_coeffs=num_filter_coeffs)
        
        # create synths

        self.harmonic_oscillator = HarmonicOscillator(sr=sr, hop_length=hop_length)
        
        self.filtered_noise = FilteredNoise(hop_length=hop_length)

        if self.use_reverb:
            self.trainable_reverb = TrainableReverb()
            self.config['outputs'] = ["reverb_audio"] # the final model output is set to the reverb audio

        print("Updated Harmonic Oscillator - Normalize after Antialiasing.")

    def forward(self, f0, loudness):
        """
        Process input f0, loudness (+ timbre eventually) with decoder, outputs synthesized audio.

        Parameters
        ----------
        f0: tensor (batch, frames)
            f0 labels
        loudness: tensor (batch, frames)
            f0 labels

        Returns
        ----------
        a : dict
            a['audio'] : synthesized audio (batch, samples)
            a['harmonic'] : synthesized harmonic part (batch, samples)
            a['noise'] : synthesized noise part (batch, samples)
            
        """

        params = self.decoder(f0, loudness)

        harmonic = self.harmonic_oscillator(params)

        noise = self.filtered_noise(params)

        audio = harmonic + noise

        if self.use_reverb:
            reverb_audio = self.trainable_reverb(audio)

        a = {}
        a['audio'] = audio
        a['harmonic'] = harmonic
        a['noise'] = noise
        if self.use_reverb:
            a['reverb_audio'] = reverb_audio

        # Merge a with params dictionary so we can access the intermediate params when evaluating
        a = {**params, **a}

        return a