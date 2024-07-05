# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch nn module for tab estimation

from midi_synth.audio_feature_transform import AudioFeatureTransform
import torch.nn as nn
import torchaudio.transforms
from util import crop_or_pad
from tab_estimation.cross_entropy_by_string import cross_entropy_by_string
from globals import *

class TabEstimator(nn.Module):
    """
    Torch nn module for tab estimation that takes in audio 
    and predicts tablature (fret positions per string).
    """
    def __init__(self,
                sr=SR,
                hop_length=HOP_LENGTH,
                input_audio_feature=TAB_ESTIMATOR_INPUT_AUDIO_FEATURE,
                nn_model_version=TAB_ESTIMATOR_MODEL_VERSION,
                eval_dict=TAB_ESTIMATOR_EVAL_DICT,
                input_audio_tag=TAB_ESTIMATOR_INPUT_AUDIO_TAG):
        """
        Initialize tab estimator module.

        Parameters
        ----------
        sr : float
            audio sampling rate
        hop_lenth : int
            samples to hop between frame starts
        input_audio_feature : str
            which feature to compute on input audio.
        nn_model_version : str
            which nn model to use ("basic-gru", will add others)
        eval_dict : dictionary
            contains flags for which evaluations to carry out
        target_audio : str
            which audio to use as input (mix_audio or mic_audio
        """
        super().__init__()      

        ####### All models should have the following defined ########

        self.name = "Tab Estimator"

        # define keys to access inputs, outputs + labels (targets) of the network
        self.config = {'inputs' : [input_audio_tag], # input: mix/mic audio 
                        'labels' : ["tab"], # compare against ground truth tab
                        'outputs' : ["tab"], # model outputs estimated tab              
        }

        # dictionary of evaluations to include in checkpoint
        self.eval_dict = eval_dict

        # set loss function
        self.loss_function = cross_entropy_by_string

        ##############################################################

        self.sr = sr
        self.hop_length = hop_length

        self.input_audio_feature = input_audio_feature

        self.audio_feature_transform = AudioFeatureTransform(mode=input_audio_feature,
                                                            sr=sr,
                                                            hop_length=hop_length)
        
        self.input_audio_feature_size = self.audio_feature_transform.feature_size

        # self.nn_model = BasicGRU()

    def forward(self, audio):
        """
        Process a conditioning tensor (and optionally audio) with 6 DDSP Mono Synths.

        Parameters
        ----------
        audio: (batch, samples)
            input guitar audio to be transcribed

        Returns
        ----------
        outputs : dict
            a['tablature'] : output audio (batch, samples)
        """

        # calculate feature from input audio
        audio_feat = self.audio_feature_transform(audio) # (batch, frames, features)

        # # send audio features thru model
        # model_output = self.nn_model(audio_feat)

        # estimate tab from model output (dense layer for fret classes? note-ons?)
        # how to handle fret classes (predict with many classes, strategy to winnow down to one class per fret, output both)
        # how to handle note on prediction?

        outputs = {}
        # outputs['tab'] = tab # (batch, samples)

        return outputs
    


