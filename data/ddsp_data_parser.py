# Author: Andy Wiggins <awiggins@drexel.edu>
# Class for creating a monophonic DDSP dataset from a folder of audio files

import librosa
import numpy as np
from globals import *
import data.guitarset_preprocessing as gset_proc
import os
from tqdm import tqdm, tqdm_notebook
from util import crop_or_pad, chunk_arr

class DDSPDataParser:
	"""
	Class for taking a folder of audio files and creating mono DDSP datasets from it.
	"""
	def __init__(self, data_home=None):
		"""
		Initialize parameters for loading guitarset.
		
		Parameters
		----------
		data_home : string
			Path to the directory containing the audio files
		"""
		self.data_home = data_home
		
		self.audio_filenames = os.listdir(data_home)


	def get_audios_and_annos_from_track(self, track_path, sr=SR, frame_rate=FRAME_RATE, item_dur=ITEM_DUR):
		"""
		Given a track path and duration (and sampling and frame rates), compute arrays of audio, freqs, and loudness.

        Parameters
        ----------
		track_path : string
        	file path to mp3 track
		sr : float
        	sample rate to load audio
		frame_rate : float
        	desired frame rate for annotations
		item_dur : float
			desired duration in seconds for items in the dataset

        Returns
        ----------
        audio_arr : numpy array (items, samples)
        	Audios from track
		freq_arr : numpy array (items, frames)
        	F0 annotations of track from CREPE F0 estimator
		loudness_arr : numpy array (items, frames)
        	Loudnesses of audio from track
	    """
		audio, rate = librosa.load(track_path, sr=sr)
		
		freqs = gset_proc.crepe_compute_f0(audio, sr=sr, frame_rate=frame_rate)
		
		audio_chunk_size = item_dur * sr
		anno_chunk_size = item_dur * frame_rate
		audio_arr = chunk_arr(audio, audio_chunk_size)
		
		freq_arr = chunk_arr(freqs, anno_chunk_size)
		
		loudness_arr = gset_proc.extract_loudness(audio_arr, sr=sr, hop_length=int(round(sr/frame_rate)), desired_len=freq_arr.shape[1])
		
    	# make sure they all have the same number of items
		
		num_items = min(audio_arr.shape[0], freq_arr.shape[0], loudness_arr.shape[0])
		audio_arr = audio_arr[:num_items, :]
		freq_arr = freq_arr[:num_items, :]
		loudness_arr = loudness_arr[:num_items, :]
		
		return audio_arr, freq_arr, loudness_arr

	def create_ddsp_dataset(self, sr=SR, frame_rate=FRAME_RATE, item_dur=ITEM_DUR):
		"""
		Given a list of track IDs, string and duration, compute arrays of audio, freqs, and loudness 

        Parameters
        ----------
		sr : float
        	sample rate to load audio
		frame_rate : float
        	desired frame rate for annotations
		item_dur : float
			desired duration in seconds for items in the dataset

        Returns
        ----------
        dataset : dict of numpy arrays
        	['audio'] : (items, samples)
			['f0'] : (items, frames)
			['loudness'] : (items, frames)
        """
		dataset = {}

		audios = []
		freqs = []
		loudnesses = []

		for audio_file in tqdm_notebook(self.audio_filenames):
			track_path = os.path.join(self.data_home, audio_file)
			audio_arr, freq_arr, loudness_arr = self.get_audios_and_annos_from_track(track_path, sr=sr, frame_rate=frame_rate, item_dur=item_dur)
			audios.append(audio_arr)
			freqs.append(freq_arr)
			loudnesses.append(loudness_arr)

		audios = np.concatenate(audios)
		freqs = np.concatenate(freqs)
		loudnesses = np.concatenate(loudnesses)

		dataset['audio'] = audios
		dataset['f0'] = freqs
		dataset['loudness'] = loudnesses

		return dataset
	

	def save_dataset(self, dataset, name, save_path=DATASETS_PATH):
		"""
		Save a dataset (dict of numpy arrays) 

        Parameters
        ----------
		dataset : dictionary of numpy arrays
        	dataset to be saved
        name : string
			name of dataset
		save_path : string
			path of where to save. Will save as save_path/name

        """	
		file = os.path.join(save_path, name)
		if not os.path.exists(save_path):
			os.mkdir(save_path)
		np.savez(file, **dataset) # ** unpacks dictionary as separate keyword arguments


