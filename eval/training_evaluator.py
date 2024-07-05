# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch model evaluator

from torch.utils.data import DataLoader
import torch.optim
import torch
from tqdm import tqdm, tqdm_notebook
import os
import datetime
from util import create_dir, torch_to_numpy
from ui import list_selection_menu
from save_load import save_batch_of_audio_from_data_dict, save_training_loss_plot, save_losses_plot, save_reverb_plot, save_spec_plots_from_batches, save_harmonic_plots
import matplotlib.pyplot as plt
import soundfile as sf
from globals import *


class TrainingEvaluator:
    """
    Class for evaluating a model.
    """
    def __init__(self, trainer=None, model=None):
        """
        Initialize model evaluator. If given a trainer, will by default use the model and dataloaders associated with the trainer. if trainer is None, a model and train and test data_loaders should be provided

        Parameters
        ----------
        trainer : model_trainer
            trainer object being used
        model : pytorch model
            model to be trained
        """
        if trainer is not None:
            self.trainer = trainer
            self.model = trainer.model
        else:
            self.trainer = None
            self.model = model

        # store the current input and output dicts so you only have to compute it once.
        self.curr_inputs = {}
        self.curr_outputs = {}

    def send_batch_thru_model(self, data_loader, dataset_name, num_items=ITEMS_PER_CHECKPOINT_EVAL):
        """
        Sends a batch of data thru the model. Num items can be provided to do smaller than a batch.

        Parameters
        ----------
        data_loader : DataLoader
            Loader to get batch of data from
            The dictionaries at key 'inputs' will be unpacked and plugged into model
        dataset_name : string
            "train" or "test", the name to store the data with
        num_items : int (< batch size)
            Number of items to crop batch to if we want fewer items than in a batch

        Returns
        ----------
        batch_dict : dict
            dictionary of batch to be input into the model
        batch_output_dict : dict
            dictionary of model outputs

        """
        #train items
        batch_dict = next(iter(data_loader))
        batch_dict = {key: batch_dict[key][:num_items] for key in batch_dict} # crop each array in the dict to the desired # of items
        batch_inputs = self.trainer.data_from_config_key(batch_dict, "inputs")
        batch_output_dict = self.model(*batch_inputs)
        self.curr_inputs[dataset_name] = batch_dict
        self.curr_outputs[dataset_name] = batch_output_dict
        # return batch_dict, batch_output_dict
    
    def get_and_save_audios(self, dataset_name, save_path, dir_name, dict_key='audio'):
        """
        Saves input (target) and output audios from batch a given DataLoader.

        Parameters
        ----------
        dataset_name : string
            "train" or "test", the name to store the data with
        save_path : string
            path to save to
        dir_name : string
            directory name to create and save within
        dict_key : string
            key to access data dicts to get audio
        """
        input = self.curr_inputs[dataset_name]
        output = self.curr_outputs[dataset_name]
        # input, output = self.send_batch_thru_model(data_loader)
        curr_path = os.path.join(save_path, dir_name)
        create_dir(curr_path)
        target = self.model.config['labels'][0] # the label in the model config is the target ex: mic_audio for hex_synth with reverb
        save_batch_of_audio_from_data_dict(input, "target", curr_path, target)
        save_batch_of_audio_from_data_dict(output, "resynth", curr_path, dict_key)

    def get_and_save_spec_plots(self, dataset_name, save_path, dir_name):
        """
        Saves input (target) and output audios from batch a given DataLoader.

        Parameters
        ----------
        dataset_name : string
            "train" or "test", the name to store the data with
        save_path : string
            path to save to
        dir_name : string
            directory name to create and save within
        """
        input = self.curr_inputs[dataset_name]
        output = self.curr_outputs[dataset_name]
        curr_path = os.path.join(save_path, dir_name)
        create_dir(curr_path)
        target_name = self.model.config['labels'][0] # the label in the model config is the target ex: mic_audio for hex_synth with reverb
        resynth_name = self.model.config['outputs'][0] # the outputs in the model config is the target ex: "audio"
        target_batch = input[target_name]
        synth_batch = output[resynth_name]
        save_spec_plots_from_batches(target_batch, synth_batch, curr_path)

    def save_harmonic_plots_for_dataset(self, dataset_name, save_path, dir_name):
        """
        Saves input (target) and output audios from batch a given DataLoader.

        Parameters
        ----------
        dataset_name : string
            "train" or "test", the name to store the data with
        save_path : string
            path to save to
        dir_name : string
            directory name to create and save within
        """
        output_batch = self.curr_outputs[dataset_name]
        curr_path = os.path.join(save_path, dir_name)
        create_dir(curr_path)
        save_harmonic_plots(output_batch, curr_path)

    def checkpoint_eval(self, save_path=None):
        """
        Evaluates a ddsp mono synth according to the provided eval_dict, saving off any necessary files.

        Parameters
        ----------
        save_path : string
            path to save to
            if none, will not save
        """
        eval_dict = self.model.eval_dict

        plt.ioff() # turn interactive mode off (don't display the plots)

        # send audio through model
        self.send_batch_thru_model(self.trainer.train_dataloader, "train")
        self.send_batch_thru_model(self.trainer.test_dataloader, "test")

        if save_path is None:
            print("save path not provided to evaluator. not evaluating")
            return

        if 'training_loss_plot' in eval_dict and eval_dict['training_loss_plot']:
            save_training_loss_plot(self.trainer, save_path)
        
        if 'losses_plot' in eval_dict and eval_dict['losses_plot']:
            save_losses_plot(self.trainer, save_path)

        if 'reverb_plot' in eval_dict and eval_dict['reverb_plot']:
            save_reverb_plot(self.model.reverb, save_path)

        if 'spec_plots' in eval_dict and eval_dict['spec_plots']:
            self.get_and_save_spec_plots("train", save_path, "train_specs")
            self.get_and_save_spec_plots("test", save_path, "test_specs")

        if 'harmonic_plots' in eval_dict and eval_dict['harmonic_plots']:
            self.save_harmonic_plots_for_dataset("train", save_path, "train_harmonic_plots")
            self.save_harmonic_plots_for_dataset("test", save_path, "test_harmonic_plots")
        
        if 'audio' in eval_dict and eval_dict['audio']:
            self.get_and_save_audios("train", save_path, "train_audio")
            self.get_and_save_audios("test", save_path, "test_audio")

        if 'harmonic_audio' in eval_dict and eval_dict['harmonic_audio']: 
            self.get_and_save_audios("train", save_path, "train_harmonic", dict_key="harmonic")
            self.get_and_save_audios("test", save_path, "test_harmonic", dict_key="harmonic")

        if 'noise_audio' in eval_dict and eval_dict['noise_audio']: 
            self.get_and_save_audios("train", save_path, "train_noise", dict_key="noise")
            self.get_and_save_audios("test", save_path, "test_noise", dict_key="noise")

        if 'reverb_audio' in eval_dict and eval_dict['reverb_audio']: 
            self.get_and_save_audios("train", save_path, "train_reverb_audio", dict_key="reverb_audio")
            self.get_and_save_audios("test", save_path, "test_reverb_audio", dict_key="reverb_audio")

        if 'residual_audio' in eval_dict and eval_dict['residual_audio'] and USE_RESIDUAL_SYNTH: 
            self.get_and_save_audios("train", save_path, "train_residual_audio", dict_key="residual_audio")
            self.get_and_save_audios("test", save_path, "test_residual_audio", dict_key="residual_audio")


            


