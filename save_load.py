# Author: Andy Wiggins <awiggins@drexel.edu>
# Functions for saving and loading

import os
import torch
import soundfile as sf
from ui import list_selection_menu
from globals import *
from util import torch_to_numpy
from plot_listen.plot import plot_training_loss, plot_losses, plot_reverb, plot_2_specs, plot_harmonic_params
import matplotlib.pyplot as plt

def load_weights_from_file(output_path=OUTPUT_PATH, experiment_name=None, checkpoint_name=None, checkpoint_path=None):
    """
    Loads and returns model weights.

    Parameters
    ----------
    output_path : str
        path to model outputs
    experiment_name : str
        name of the experiment (and its dir)
    checkpoint_name : str
        name of the checkpoint (for instance "25 epochs")  

    Returns
    ----------
    state_dict : dictionary
        model state dict containing weights    
    """

    if checkpoint_path is not None:
        load_path = checkpoint_path
    else:
        load_path = output_path
        if experiment_name is None:
            # To Do: sort by date and show n top
            list_avail_exp_names = os.listdir(output_path)
            experiment_name = list_selection_menu(list_avail_exp_names)   
        load_path = os.path.join(load_path, experiment_name, "checkpoints")
        if checkpoint_name is None:
                # To Do: sort by date and show n top
            list_avail_cp_names = os.listdir(load_path)
            checkpoint_name = list_selection_menu(list_avail_cp_names)   
        load_path = os.path.join(load_path, checkpoint_name, "model_checkpoint.pt")

    state_dict = torch.load(load_path)['model_state_dict']

    return state_dict

def save_audio(audio, path, sr=SR):
    """
    Saves audio given a file path and its sample rate

    Parameters
    ----------
    audio : 1D numpy array or tensor
        path to model outputs
    path : string
        file to save to
    sr : float
        sampling rate    
    """
    if torch.is_tensor(audio):
        audio = torch_to_numpy(audio)
    sf.write(path, audio, sr)

def save_batch_of_audio_from_data_dict(data_dict, data_name, save_path, dict_key='audio', sr=SR):
    """
    Takes in a dictionary of data and saves to wav any audio found under the key 'audio' 

    Parameters
    ----------
    data_dict : dict
        dictionary of data to extract and save audio from
        example: data_dict['audio'] : shape = (batch, samples)
    data_name : string
        name will be incorporated into save filename
    save_path : string
        path to save extracted audio to
    dict_key : string
        key to access data_dict at and get audio
    sr : float
        sampling rate

    Returns
    ----------
    batch_output_dict : dict
        dictionary of model outputs
    """
    if dict_key not in data_dict:
        return # can't save audio if the desired key is not present in the data dict

    audios = data_dict[dict_key] # default key is 'audio'
    for i, audio in enumerate(audios):            
        filename = os.path.join(save_path, f"{i}_{data_name}.wav")
        save_audio(audio, filename, sr)

def save_training_loss_plot(trainer, save_path):
    """
    Given a model trainer object and a save path, save the training loss plot

    Parameters
    ----------
    trainer : ModelTrainer
        trainer to plot training loss for
    save_path : string
        path to save plot to
    """
    plot_training_loss(trainer)
    plt.savefig(os.path.join(save_path, "training_loss.png"))
    plt.close()

def save_losses_plot(trainer, save_path):
    """
    Given a model trainer object and a save path, save the losses plot

    Parameters
    ----------
    trainer : ModelTrainer
        trainer to plot losses for
    save_path : string
        path to save plot to
    """
    plot_losses(trainer)
    plt.savefig(os.path.join(save_path, "losses.png"))
    plt.close()

def save_reverb_plot(reverb, save_path):
    """
    Given a model trainer object and a save path, save the losses plot

    Parameters
    ----------
    reverb : TrainableReverb
        reverb module to make plots of
    save_path : string
        path to save plot to
    """
    plot_reverb(reverb)
    plt.savefig(os.path.join(save_path, "reverb.png"))
    plt.close()

def save_spec_plots_from_batches(target_batch, synth_batch, save_path):
    """
    Given a model trainer object and a save path, save the losses plot

    Parameters
    ----------
    target_batch : tensor (batches, samples)
        batch of target audios to plot
    synth_batch : tensor (batches, samples)
        batch of synthesized audios to plot
    save_path : string
        path to save plots to
    """

    for i in range(target_batch.shape[0]):

        filename = os.path.join(save_path, f"specs_{i}.png")
        
        target_audio = target_batch[i]
        synth_audio = synth_batch[i]
        plot_2_specs(target_audio,synth_audio)
        plt.savefig(filename)
        plt.close()
        
        return  # only plot 1st clip for now, it takes a while to plot these
    
def save_harmonic_plots(p, save_path):
    """
    Given a dict of a batch of harmonic params, save plots of the effective params

    Parameters
    ----------
    p : dict
        p['f0'] : f0 envelope tensor (batch, frames)
        p['overall_amplitude'] : overall amplitude (batch, frames)
        p['harm_distr'] : harmonic distribution (sums to 1) tensor (batch, # harmonics, frames)

    idx : int
        index in batch to plot example from
    """

    batch = p['f0'].shape[0]

    for i in range(batch):

        filename = os.path.join(save_path, f"harmonic_params_{i}.png")
        
        plot_harmonic_params(p, idx=i)
        plt.savefig(filename)
        plt.close()
