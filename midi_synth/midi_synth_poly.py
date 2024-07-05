# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch synth module with midi-like conditioning, predicting string params in unison

from ddsp.harmonic_oscillator import HarmonicOscillator
from ddsp.filtered_noise import FilteredNoise
from ddsp.trainable_reverb import TrainableReverb
from ddsp.multi_scale_spectral_loss import multi_scale_spectral_loss
from midi_synth.midi_util import midi_to_hz
from midi_synth.audio_feature_transform import AudioFeatureTransform
from midi_synth.polyphonic_net import PolyphonicNet
import torch.nn as nn
import torchaudio.transforms
import save_load
from util import crop_or_pad
from globals import *

class MidiSynthPoly(nn.Module):
    """
    Torch nn module for a ddsp-style synth that takes in conditioning in a midi-like format
    and synthesizes audio by predicting polyphonic params (all strings at once).
    """
    def __init__(self,
                sr=SR,
                hop_length=HOP_LENGTH,
                num_oscillators=NUM_OSCILLATORS,
                num_filter_coeffs=NUM_FILTER_COEFFICIENTS,
                reverb_length=REVERB_IR_LENGTH,
                input_audio_feature=MIDI_SYNTH_POLY_INPUT_AUDIO_FEATURE,
                poly_net_model_version=POLY_NET_MODEL_VERSION,
                eval_dict=MIDI_SYNTH_POLY_EVAL_DICT,
                target_audio=MIDI_SYNTH_TARGET_AUDIO):
        """
        Initialize midi synth module.

        Parameters
        ----------
        sr : float
            audio sampling rate
        hop_lenth : int
            samples to hop between frame starts
        num_oscillators : int
            number of oscillators to get harm distribution for (corresponds to number of oscillators in synth)
        filter_coeffs : int
            number of filter coefficients to output for use in filtered noise synth 
        reverb_length : int
            length of reverb impulse response in samples
        input_audio_feature : str
            which feature to compute on input audio. If None, input audio not used
        poly_net_model_version : str
            which nn model to use in the poly network ("basic-gru", will add others)
        eval_dict : dictionary
            contains flags for which evaluations to carry out
        target_audio : str
            which audio to use as a ground truth (mix_audio or mic_audio
        """
        super().__init__()      

        ####### All models should have the following defined ########

        self.name = "MIDI Synth (Poly)"

        # define keys to access inputs, outputs + labels (targets) of the network
        self.config = {'inputs' : ["conditioning"], 
                        'labels' : [target_audio], # compare against, mix/mic audio
                        'outputs' : ["audio"], # model outputs synthesized audio               
        }

        # dictionary of evaluations to include in checkpoint
        self.eval_dict = eval_dict

        # set loss function
        self.loss_function = multi_scale_spectral_loss

        ##############################################################

        self.sr = sr
        self.hop_length = hop_length

        self.input_audio_feature = input_audio_feature
        if self.input_audio_feature is not None:
            print(f"Using {self.input_audio_feature} as additional input to synth.")
            self.config['inputs'] += [target_audio] # add "mic_audio" to inputs, for instance, so that mfcc can be calculated

            self.audio_feature_transform = AudioFeatureTransform(mode=input_audio_feature,
                                                                sr=sr,
                                                                hop_length=hop_length)
            self.input_audio_feature_size = self.audio_feature_transform.feature_size
        else:
            self.input_audio_feature_size = 0

        self.polyphonic_net = PolyphonicNet(model_version=poly_net_model_version,
                                            num_oscillators=num_oscillators,
                                            num_filter_coeffs=num_filter_coeffs,
                                            audio_feature_size=self.input_audio_feature_size
        )
       
        self.harmonic_oscillator = HarmonicOscillator(sr=sr, hop_length=hop_length, inharmonicity=INHARMONICITY)

        self.filtered_noise = FilteredNoise(hop_length=hop_length)
        
        self.reverb = TrainableReverb(reverb_length=reverb_length)

    def forward(self, conditioning, input_audio=None, output_predicted_params=MIDI_SYNTH_OUTPUT_PREDICTED_PARAMS, output_factored_audio=MIDI_SYNTH_OUTPUT_FACTORED_AUDIO):
        """
        Process a conditioning tensor (and optionally audio) with 6 DDSP Mono Synths.

        Parameters
        ----------
        conditioning: tensor (batch, frames, num_strings=6, pitch/vel=2)
            pitch and onset velocity labels across 6 strings
        input_audio: tensor (batch, samples)
            target audio to be synthesized (for timbre info when combined with tab network)
        output_predicted_params: bool
            if true, output dict contains params predicted by polyphonic net
        output_factored_audio: bool
            if true, output dict contains harmonic, noise, dry, and individual_string audio

        Returns
        ----------
        outputs : dict
            a['audio'] : output audio (batch, samples)
        """
        # get num_frames
        batch = conditioning.shape[0]
        num_frames = conditioning.shape[1]

        # normalize conditioning
        conditioning = conditioning / MIDI_NORM # scale conditioning in [0,1]

        # calculate feature from input audio
        if self.input_audio_feature is not None and input_audio is not None:
            audio_feat = self.audio_feature_transform(input_audio, num_frames=num_frames) # give num frames to crop or pad to matching length
        else:
            audio_feat = None

        params = self.polyphonic_net(conditioning, audio_feat) # params each have first dim = 6 x batch

        # set up string indices (to indicate string num to oscillator, so that we can have different inharmonicity per string)
        string_idxs = []
        for i in range(6):
            string_idxs.append(torch.IntTensor([i]).expand(batch).to(conditioning.device)) # (batch)
        string_idx = torch.stack(string_idxs, dim=0) # (6, batch)
        string_idx = string_idx.flatten().to(conditioning.device) # (batch x 6)     

        harmonic = self.harmonic_oscillator(params, string_idx=string_idx) # (batch x 6, samples)

        noise = self.filtered_noise(params) # (batch x 6, samples)

        string_audio = harmonic + noise # (batch x 6, samples)

        # revert shape back to (6, batch, samples)
        string_audio = torch.unflatten(string_audio, 0, (6, batch))
        # sum string audios
        dry_audio = torch.sum(string_audio, dim=0) # (batch, samples)

        audio = self.reverb(dry_audio)

        outputs = {}
        outputs['audio'] = audio # (batch, samples)

        # optionally, save off predicted params (per string)
        if output_predicted_params:

            # overall amplitude
            outputs['overall_amplitude'] = torch.unflatten(params['overall_amplitude'], 0, (6, batch))
            # harmonic distribution
            outputs['harm_distr'] = torch.unflatten(params['harm_distr'], 0, (6, batch))
            # H
            outputs['H'] = torch.unflatten(params['H'], 0, (6, batch))
            # f0
            outputs['f0'] = torch.unflatten(params['f0'], 0, (6, batch))

        if output_factored_audio:
            # string audio
            outputs['string_audio'] = string_audio # already been unflattened above (6, batch, samples)
            # harmonic
            outputs['harmonic'] = torch.unflatten(harmonic, 0, (6, batch)) # (6, batch, samples)
            # noise
            outputs['noise'] = torch.unflatten(noise, 0, (6, batch)) # (6, batch, samples)
            # noise
            outputs['dry_audio'] = dry_audio # (batch, samples)

        return outputs
    


