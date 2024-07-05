# Author: Andy Wiggins <awiggins@drexel.edu>
# global variables to be accessed from all files

import torch
import torch.nn as nn
import torch.optim
import numpy as np
import datetime

####### General Audio #######
SR = 16000
HOP_LENGTH = 64 # in samples (4ms)
FRAME_RATE = SR / HOP_LENGTH # in frames/sec
N_FFT = 2048
ITEM_DUR = 3.0 # training item duration (in seconds), default: 1.0, MIDI: 3.0, violin: 4.0

####### Guitar Tab #######
GUITAR_STRING_LETTERS = ("E", "A", "D", "G", "B", "e")
NUM_FRETS = 19
NUM_FRET_CLASSES = NUM_FRETS + 2 # for open string, closed
OPEN_STRING_MIDI_PITCHES = (40, 45, 50, 55, 59, 64)
PITCH_BINS_PER_SEMITONE = 5
NUM_QUANTIZED_PITCHES_PER_STRING = ((NUM_FRETS + 1) * PITCH_BINS_PER_SEMITONE) + 1 # inner + 1 is for open (gets multiplied), outer is for closed


####### Numerical #######
EPS = 1e-7

####### Training #######
BATCH_SIZE = 6 # standard: 16, premium: 64, premium-3sec-items: 32, standard-3sec-items: 6
NUM_WORKERS = 4 # standard: 4, premium: 8 # use multiple threads to ready the next batch in parallel
TRAIN_SHUFFLE = True
OPTIMIZER = torch.optim.Adam
LEARNING_RATE = 0.001*0.1 # default: 0.001
NUM_EPOCHS = 2000 # default 20000
EPOCHS_PER_CHECKPOINT = 4 # default: 20
EPOCHS_PER_CHECKPOINT_RETAINED = 20 # these checkpoints aren't deleted # default: 100
EPOCHS_BEFORE_SAVING_BEST_CHECKPOINTS = 5 # after training this many epochs, trainer will save a checkpoint when a minimum validation loss is reached
LEARNING_RATE_STEP_SIZE = 50 # in epochs # default: 50
LEARNING_RATE_DECAY = 1.0 # recent default: 0.98, prev default: 1.0

###### guitarist-net Cloud Storage Path ######
GUITARIST_NET_STORAGE_PATH = "/content/drive/MyDrive/Research/guitarist-net-storage/" # for wiggins@excitecenter.org
# GUITARIST_NET_STORAGE_PATH = "/content/drive/MyDrive/Andy/guitarist-net-storage/" # for colab@excitecenter.org 

####### GuitarSet Original Settings #######
GSET_PATH = GUITARIST_NET_STORAGE_PATH + "data-source/GuitarSet"
GSET_SR = 44100
GSET_HOP_LENGTH = 256
GSET_FRAME_RATE = GSET_SR / GSET_HOP_LENGTH # in frames/sec
USE_CREPE_F0_LABELS = False
CREPE_CONFIDENCE_THRESH = 0.5

####### Loading and Saving #######
DATASETS_PATH = GUITARIST_NET_STORAGE_PATH + "datasets"
OUTPUT_PATH = GUITARIST_NET_STORAGE_PATH + "output"

####### GPU #######
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####### Data Types #######
DEFAULT_TORCH_DTYPE = torch.float32
DEFAULT_NP_DTYPE = np.float32

####### DDSP Synths#######
NUM_OSCILLATORS = 101
NUM_FILTER_COEFFICIENTS = 65
INHARMONICITY =  "per-string" # None to turn off, "global" for 1 param, "per-string" for 6 params
INHARM_BETA_INIT = 0.0
PER_STRING_INHARM_BETA_INIT = [ 1.07e-4,
                                6.02e-5,
                                5.54e-5,
                                5.325e-5,
                                5.11e-5,
                                2.09e-5,
    # These values are the measure beta values (post sigmoid!)
    # From this paper: https://publicwebuploads.uwec.edu/documents/Musical-string-inharmonicity-Chris-Murray.pdf
    # The paper uses electric guitar so I changed/approximated the G-string beta based on it being another thing wound string on acoustic guitar
]
FREEZE_INHARM_PARAMS = False # Default: TRue
REVERB_BYPASS = False # If true reverb is bypassed. Default: False
DDSP_MONO_SYNTH_USE_REVERB = True # default: True
REVERB_IR_LENGTH = SR # Default: SR (1 second, in samples)

####### DDSP DECODER #######
## MLP ##
MLP_NUM_UNITS = 512 # nodes per layer
MLP_NUM_LAYERS = 3
MLP_ACTIVATION = torch.nn.LeakyReLU
MLP_INPUT_SIZE = 1
## GRU ##
GRU_NUM_UNITS = 512

####### Multi-scale Spectral Loss #######
NUM_SPECTRAL_LOSSES = 6
MIN_SPECTRAL_LOSS_FFT_SIZE = 64
SPECTRAL_LOSS_LOG_WEIGHTING = 1.0 # Default: 1.0
SPECTRAL_LOSS_LINEAR_WEIGHTING = 1.0 # Default: 1.0
USE_REGULARIZATION_LOSS = False
REGULARIZATION_LOSS_ORDER = 1 # 1 for L1, 2 for L2, etc.
REGULARIZATION_LOSS_WEIGHT = 0.01 # default: 0.01
NORMALIZE_AUDIO_IN_LOSS_FUNCTION = False # default: False
NORMALIZE_IN_LOSS_STFT = True # default: True
LOSS_FUNCTION_TREBLE_BOOST_FRAC = 0.0 # perctange of FFT bins, Default: 0.0
LOSS_FUNCTION_TREBLE_BOOST_AMT = 1.0 # multiplier for FFT bins, Default: 1.0

####### Synth Improvement #######
IMPROVEMENT_LAYERS_HIDDEN_SIZE = 512
IMPROVEMENT_LAYERS_NUM_LAYERS = 3

####### Time Zone #######
TIME_ZONE_HOURS = -4

####### Training Evaluator #######
ITEMS_PER_CHECKPOINT_EVAL = 10 # must be less than batch size
DDSP_MONO_SYNTH_EVAL_DICT = {   'training_loss_plot' : False,
                                'losses_plot' : True,
                                'audio' : True,
                                'harmonic_audio' : True,
                                'harmonic_plots' : True,
                                'noise_audio' : True,
                                'reverb_audio' : DDSP_MONO_SYNTH_USE_REVERB,
                                'spec_plots' : True,
                            }

####### DDSP HEX SYNTH #######
DDSP_HEX_SYNTH_USE_REVERB = True  # at the end on the overall audio
DDSP_HEX_SYNTH_EVAL_DICT = {   'training_loss_plot' : False,
                                'losses_plot' : True,
                                'audio' : True,
                                'harmonic_audio' : False,
                                'noise_audio' : False,
                                'reverb_audio' : DDSP_MONO_SYNTH_USE_REVERB
                            }
DDSP_HEX_SYNTH_TARGET_AUDIO = "mic_audio"
HEX_SYNTH_LOAD_MONO_SYNTH_WEIGHTS = True
HEX_MONO_SYNTH_WEIGHT_PATHS = { 'E' : "/content/drive/MyDrive/Research/guitarist-net-storage/output/2022-09-13 02:28PM DDSP Mono Synth : E string/checkpoints/1825 epochs/model_checkpoint.pt",
                                'A' : "/content/drive/MyDrive/Research/guitarist-net-storage/output/2022-09-12 12:18PM DDSP Mono Synth : A string/checkpoints/1225 epochs/model_checkpoint.pt",
                                'D' : "/content/drive/MyDrive/Research/guitarist-net-storage/output/2022-09-02 06:10PM DDSP Mono Synth : D string/checkpoints/2675 epochs/model_checkpoint.pt",
                                'G' : "/content/drive/MyDrive/Research/guitarist-net-storage/output/2022-09-02 12:19AM DDSP Mono Synth : G string/checkpoints/1675 epochs/model_checkpoint.pt",
                                'B' : "/content/drive/MyDrive/Research/guitarist-net-storage/output/2022-09-01 06:27PM DDSP Mono Synth : B string/checkpoints/900 epochs/model_checkpoint.pt",
                                'e' : "/content/drive/MyDrive/Research/guitarist-net-storage/output/2022-09-01 05:32PM DDSP Mono Synth : e string/checkpoints/2500 epochs/model_checkpoint.pt",
                            }


####### GuitarSet MIDI-like dataset #######
# note: the following ranges are estimated based on loudness only at time of onset
GSET_MIN_STRING_LOUDNESS_BOUND = -60
GSET_MAX_STRING_LOUDNESS_BOUND = 20
MIDI_NORM = 128
OMIT_INCOMPLETE_CHUNKS = True # If true, when chunking data into items, don't include items that don't span the full duration.
MIDI_DATASET_USE_LOUDNESS_ENVELOPE = False # otherwise, the midi dataset just uses onset velocity (closer to actual MIDI). This parameter only has an effect during dataset creation.

####### Mono Network String Embedding #######
STRING_EMBEDDING_SIZE = 6
STRING_EMBEDDING_NUM = 6

####### Mono Network String Masking #######
MASK_STRING_DATA_WITH_CONDITIONING = False # if True, zero out predicted string params if the conditioning on that string is zero.
EXTEND_STRING_MASK = True
STRING_MASK_LEFT_EXTEND_FRAMES = 3 # Default: 3 let strings sound before the conditioning is nonzero
STRING_MASK_RIGHT_EXTEND_FRAMES = 0 # Default: 0 let strings sound after the conditioning is nonzero

####### Midi Synth #######
MIDI_SYNTH_EVAL_DICT = {   
                        'losses_plot' : True,
                        'reverb_plot' : True,
                        'spec_plots' : True,
                        'audio' : True,
                        'residual_audio' : True,
                        }
MIDI_SYNTH_TARGET_AUDIO = "mic_audio"
MIDI_SYNTH_USE_STRING_EMBEDDING = True # if True use string index as input to mono network, default: True
MIDI_SYNTH_USE_MFCC_INPUT = True # if True, then the audio's mfcc features are included as an input to the midi synth's mono network, default: False
MIDI_SYNTH_USE_STFT_INPUT = False # if True, then the audio's stft is included as an input to the midi synth's mono network, default: False
N_MFCC = 40
N_MELS = 128
MIDI_SYNTH_LEARN_PITCH_ADJUSTMENT = False # default: False
PARALLELIZE_OVER_STRINGS = True
USE_CONTEXT_NET = True
CONTEXT_SIZE = 32
USE_SMALL_MONO_NET = True
MONO_NET_SM_LINEAR_1_SIZE = 128*4 # default: 128
MONO_NET_SM_LINEAR_2_SIZE = 128*4 # default: 128, althought old version mistakenly had 192 that I think was not used
MONO_NET_SM_GRU_NUM_UNITS = 192*4 # default: 192
MIDI_SYNTH_OUTPUT_PREDICTED_PARAMS = True # default: True, seeing if false uses less memory
MIDI_SYNTH_OUTPUT_FACTORED_AUDIO = True # default: True, seeing if False uses less memory
MIDI_SYNTH_GAIN_PARAM = False # Default: False

####### Encodec #######
ENCODEC_RESAMPLE_CODEBOOK = False # if true, the codebook from encodec is resampled to match the frame rate of the synth.

####### Residual Synth #######
USE_RESIDUAL_SYNTH = False # If true, MIDI synth trains a 7th ddsp synth to capture random guitar noises
RESIDUAL_SYNTH_MODE = "stft" # "stft" for unconstrained spec prediciton, or "ddsp" for prediction of f0 and amplitudes
BYPASS_RESIDUAL_DURING_EVAL = False # if true, the residual audio is not used during eval, only during training.
RESIDUAL_LINEAR_1_SIZE = 64 # default: 64
RESIDUAL_LINEAR_2_SIZE = 64 # default: 64
RESIUDAL_GRU_NUM_UNITS = 64 # default: 64

####### MIDI Synth (Poly) #######
MIDI_SYNTH_POLY_INPUT_AUDIO_FEATURE = "mfcc" # future options: "stft", "oobleck", "melspec", None
MIDI_SYNTH_POLY_EVAL_DICT = {   
                        'losses_plot' : True,
                        'reverb_plot' : True,
                        'spec_plots' : True,
                        'audio' : True,
                        }

####### Polyphonic Net #######
POLY_NET_MODEL_VERSION = "basic-gru" # options: "basic-gru", "basic-transformer", "transformer-2D" 
POLY_NET_LINEAR_1_SIZE = 128
POLY_NET_LINEAR_2_SIZE = 128
POLY_NET_GRU_NUM_UNITS = 192
####### Networks #######
#----- BasicGRU --------
BASIC_GRU_LINEAR_1_SIZE = 128
BASIC_GRU_LINEAR_2_SIZE = 128
BASIC_GRU_GRU_NUM_UNITS = 192
#----- BasicTransformer --------
BASIC_TRANSFORMER_LINEAR_1_SIZE = 128
BASIC_TRANSFORMER_DIM = 192
BASIC_TRANSFORMER_NUM_HEADS = 8
BASIC_TRANSFORMER_NUM_LAYERS = 4
BASIC_TRANSFORMER_LINEAR_2_SIZE = 128
#----- Transformer2D --------
# both direction transformers are given the same size
TRANSFORMER2D_LINEAR_1_SIZE = 128
TRANSFORMER2D_DIM = 192
TRANSFORMER2D_FEAT_NUM_HEADS = 6   # must divide ITEM_DUR * FRAME_RATE (750 with default settings)
TRANSFORMER2D_TIME_NUM_HEADS = 8   # must divide linear 1 size
TRANSFORMER2D_NUM_LAYERS = 4
TRANSFORMER2D_LINEAR_2_SIZE = 128

####### TabEstimator #######
TAB_ESTIMATOR_INPUT_AUDIO_FEATURE = "melspec" # future: add cqt
TAB_ESTIMATOR_EVAL_DICT = {'losses_plot' : True} # future: add tab/pitch plots?
TAB_ESTIMATOR_MODEL_VERSION = None # future: add more model versions
TAB_ESTIMATOR_INPUT_AUDIO_TAG = "mic_audio" # which audio to use as model input