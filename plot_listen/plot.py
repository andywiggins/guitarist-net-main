# Author: Andy Wiggins <awiggins@drexel.edu>
# plotting functions

import matplotlib.pyplot as plt
from plot_listen.listen import play
import numpy as np
import torch
from util import torch_to_numpy, numpy_to_torch, safe_log
from torch import stft
from scipy.fft import fft, fftfreq
import librosa
from globals import *

def plot_single_string_item(item, sr=SR):
    """
    Show relevant data from a single string training item. 

    Parameters
    ----------
    item : dict
        single item from a SingleStringDataset
    sr : float
        sampling rate of audio

    Returns
    ----------
    audio : Audio object
        Audio playback display object
    """
    # TO DO: Make these plots look nice
    fig, axs = plt.subplots(nrows=4,figsize=(8,8))

    num_samples = len(item['audio'])
    num_frames = len(item['f0'])
    
    axs[0].plot(item['audio'])
    axs[0].set_title("audio")
    # axs[0].xticks = [0, num_samples]
    
    axs[1].plot(item['f0'])
    axs[1].set_title("f0")
    # axs[1].xticks = [0, num_frames]
    
    axs[2].plot(item['voicing'])
    axs[2].set_title("voicing")
    # axs[2].xticks = [0, num_frames]

    axs[3].plot(item['loudness'])
    axs[3].set_title("loudness")
    # axs[3].xticks = [0, num_frames]

    plt.subplots_adjust(hspace=1.5) # adjust margin between subplots

    return play(item['audio'], sr=sr)

def plot_ddsp_item(item, sr=SR):
    """
    Show relevant data from a mono ddsp training item.

    Parameters
    ----------
    item : dict
        single item from a DDSPDataset
    sr : float
        sampling rate of audio

    Returns
    ----------
    audio : Audio object
        Audio playback display object
    """
    # TO DO: Make these plots look nice
    fig, axs = plt.subplots(nrows=3,figsize=(8,8))

    num_samples = len(item['audio'])
    num_frames = len(item['f0'])
    
    axs[0].plot(item['audio'])
    axs[0].set_title("audio")
    
    axs[1].plot(item['f0'])
    axs[1].set_title("f0")

    axs[2].plot(item['loudness'])
    axs[2].set_title("loudness")

    plt.subplots_adjust(hspace=1.5) # adjust margin between subplots

    return play(item['audio'], sr=sr)

def plot(t):
    """
    Quickly plot an array or tensor. 

    Parameters
    ----------
    t : array or list (probably 1D)
        vector to be plotted

    """
    if type(t) == type(torch.ones((1,1))):
        plt.plot(torch_to_numpy(t))
    else:
        plt.plot(t)

def plot_training_loss(trainer):
    """
    Given a model trainer object, plot the training loss over the number of epochs

    Parameters
    ----------
    trainer : ModelTrainer
        trainer to plot training loss for

    """
    epochs_axis = np.array(range(trainer.epochs_trained)) + 1 # add one to start at epoch 1
    loss_vals = trainer.training_loss_values
    plt.plot(epochs_axis, loss_vals)
    plt.title("training loss")
    plt.xlabel("epochs trained")

def plot_losses(trainer):
    """
    Given a model trainer object, plot the training loss and validation loss over the number of epochs

    Parameters
    ----------
    trainer : ModelTrainer
        trainer to plot losses for

    """
    epochs_axis = np.array(range(trainer.epochs_trained)) + 1 # add one to start at epoch 1
    train_loss_vals = trainer.training_loss_values
    valid_loss_vals = trainer.validation_loss_values
    plt.plot(epochs_axis, train_loss_vals, label="training loss")
    plt.plot(epochs_axis, valid_loss_vals, label="validation loss")
    plt.title("losses")
    plt.xlabel("epochs trained")
    plt.legend(loc='best')

def plot_reverb(verb):
    """
    Given a model trainer object, plot the training loss and validation loss over the number of epochs

    Parameters
    ----------
    verb : TrainableReverb
        reverb module to plot
    """

    # extract params from reverb
    drywet = torch.sigmoid(verb.drywet).item()
    decay = verb.decay.item()

    # get fir and compute magnitude freq response
    fir = torch_to_numpy(verb.generate_fir()[0])
    mag_freq = np.abs(fft(fir))
    fft_freq = fftfreq(REVERB_IR_LENGTH, 1 / SR) # the fft freqs (x axis in mag freq plot)

    # make plot
    fig, axs = plt.subplots(nrows=2,figsize=(8,8))
    axs[0].plot(fir)
    axs[0].set_title(f"Impulse Response\nDecay: {decay:.3f}, Wet/Dry: {drywet:.3f}")
    axs[0].set_xlabel("samples")
    axs[1].plot(fft_freq, mag_freq)
    axs[1].set_title("Magnitude Frequency Response")
    axs[1].set_xlabel("Hz")
    fig.suptitle('TrainableReverb', fontsize=16)
    fig.tight_layout()

def plot_spec(audio, title="Log Magnitude Spectrogram"):
    """
    Plot log spectrogram of some audio

    Parameters
    ----------
    audio : np.array or torch tensor (samples,)
        audio data to plot
    """
    # set up plot
    fig, ax = plt.subplots(ncols=1, sharey=True, figsize=(8,4))

    # convert to torch tensors
    if not torch.is_tensor(audio):
        audio = numpy_to_torch(audio)

    ### target   
    audio_stft = stft(audio, 
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH, 
                win_length=N_FFT,
                window=torch.hann_window(N_FFT, device=audio.device), #move window to gpu
                normalized=True,
                return_complex=True).abs()
    audio_stft = safe_log(audio_stft)
    audio_stft = torch_to_numpy(audio_stft)


    ax.set_title(title)
    img = librosa.display.specshow(audio_stft, y_axis='linear', 
                                   x_axis='time', sr=SR, n_fft=N_FFT, 
                                   hop_length=HOP_LENGTH, ax=ax,
                                   vmin=-18, vmax=1)

    # Overall plot settings 
    plt.figtext(0.0, -0.1,f"Spec Range: [{np.min(audio_stft):.2f}, {np.max(audio_stft):.2f}]")
    fig.tight_layout()



def plot_2_specs(targ, resyn):
    """
    Plot spectrograms of target and synth audio

    Parameters
    ----------
    targ : np.array or torch tensor (samples,)
        audio data to plot
    resyn : np.array or torch tensor (samples,)
        audio data to plot
    """
    # set up plot
    fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(8,4))

    # convert to torch tensors
    if not torch.is_tensor(targ):
        targ = numpy_to_torch(targ)
    if not torch.is_tensor(resyn):
        resyn = numpy_to_torch(resyn)

    ### target   
    targ_stft = stft(targ, 
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH, 
                win_length=N_FFT,
                window=torch.hann_window(N_FFT, device=targ.device), #move window to gpu
                normalized=True,
                return_complex=True).abs()
    targ_stft = safe_log(targ_stft)
    targ_stft = torch_to_numpy(targ_stft)

    ### resynth  
    resyn_stft = stft(resyn, 
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH, 
                win_length=N_FFT,
                window=torch.hann_window(N_FFT, device=resyn.device), #move window to gpu
                normalized=True,
                return_complex=True).abs()
    resyn_stft = safe_log(resyn_stft)
    resyn_stft = torch_to_numpy(resyn_stft)

    ax[0].set_title(f"Target Audio")
    img = librosa.display.specshow(targ_stft, y_axis='linear', 
                                   x_axis='time', sr=SR, n_fft=N_FFT, 
                                   hop_length=HOP_LENGTH, ax=ax[0],
                                   vmin=-18, vmax=1)

    ax[1].set_title(f"Synth Audio")
    img = librosa.display.specshow(resyn_stft, y_axis='linear', 
                                   x_axis='time', sr=SR, n_fft=N_FFT, 
                                   hop_length=HOP_LENGTH, ax=ax[1],
                                   vmin=-18, vmax=1)

    # Overall plot settings 
    fig.suptitle('Log Magnitude Spectrograms', fontsize=14)
    plt.figtext(0.0, -0.1,f"Target Spec Range: [{np.min(targ_stft):.2f}, {np.max(targ_stft):.2f}]\nSynth Spec Range: [{np.min(resyn_stft):.2f}, {np.max(resyn_stft):.2f}]")
    fig.tight_layout()

    

def plot_midi_conditioning(cond):
    """
    Plot a midi conditioning array, showing the continuous pitch (midi range) and the onset velocity for each guitar string 

    Parameters
    ----------
    cond : np.array (frames, num_strings=6, pitch/velocity=2)
        single item from a SingleStringDataset
    """
    fig, axs = plt.subplots(nrows=12,figsize=(8,30))
    frames_list = range(len(cond))

    for i in range(6):
        str_i = cond[:,i,:] # conditioning for string i

        # set plot index
        plot_ind = i * 2

        # the string conditioning is nonzero, plot pitch
        if np.any(str_i):
            axs[plot_ind].plot(frames_list, str_i[:,0])
        axs[plot_ind].set_title(f"{GUITAR_STRING_LETTERS[i]}-string Pitch")
        axs[plot_ind].set_xlim(0,len(cond))
        axs[plot_ind].set_ylim(0,127)

        # set plot index
        plot_ind = (i * 2) + 1

        # the string conditioning is nonzero, plot onset vel
        if np.any(str_i):
            axs[plot_ind].stem(frames_list, str_i[:,1], use_line_collection=True)
        axs[plot_ind].set_title(f"{GUITAR_STRING_LETTERS[i]}-string Onset Velocity")
        axs[plot_ind].set_xlim(0,len(cond))
        axs[plot_ind].set_ylim(0,127)


    plt.subplots_adjust(hspace=1.5) # adjust margin between subplots


def plot_multi_stfts(x, 
                    num_losses=NUM_SPECTRAL_LOSSES, 
                    min_fft_size=MIN_SPECTRAL_LOSS_FFT_SIZE,
                    subplot_size=3):
    """
    Plot multi-scale spectrograms of some audio x, given num losses (scales), min fft size and a subplot size  

    Parameters
    ----------
    x : np.array or torch tensor (samples,)
        audio data to plot
    num_losses : int
        number of spectrograms scales to compute
    min_fft_size : int
        size of smallest fft, each size after doubles
    subplot_size : int, float
        size of each subplot
    
    """

    # # get window sizes
    fft_sizes = [min_fft_size * (2**i) for i in range(num_losses)]

    # set up plot
    figsize = (subplot_size*num_losses, subplot_size*2)
    fig, ax = plt.subplots(2,num_losses, figsize=figsize)

    # compute stft and log_stft for x
    if not torch.is_tensor(x):
        x = numpy_to_torch(x)

    # for each fft size
    for i, fft_size in enumerate(fft_sizes):
        hop_length = int(fft_size / 4)  # overlap by 75%
          
        x_stft = stft(x, 
                    n_fft=fft_size, 
                    hop_length=hop_length, 
                    win_length=fft_size,
                    window=torch.hann_window(fft_size, device=x.device), #move window to gpu
                    normalized=True,
                    return_complex=True).abs()
        x_log_stft = safe_log(x_stft)

        # convert to numpy array
        x_stft = torch_to_numpy(x_stft)
        x_log_stft = torch_to_numpy(x_log_stft)


        # Make plots
        ax[0, i].imshow(x_stft, aspect='auto', origin='lower', interpolation=None)
        ax[0, i].set_title(f"FFT size: {fft_size}")
        ax[1, i].imshow(x_log_stft, aspect='auto', origin='lower', interpolation=None)

    # Overall plot settings 
    ax[0, 0].set_ylabel("STFT", rotation=0, size='large')
    ax[1, 0].set_ylabel("Log STFT", rotation=0, size='large')
    ax[0,0].yaxis.set_label_coords(-0.4, 0.4)
    ax[1,0].yaxis.set_label_coords(-0.4, 0.4)
    fig.suptitle('Multi-Scale Spectrograms', fontsize=16)
    fig.tight_layout()

def plot_diff_multi_stfts(x, y, 
                    standard_scale=False,
                    num_losses=NUM_SPECTRAL_LOSSES, 
                    min_fft_size=MIN_SPECTRAL_LOSS_FFT_SIZE,
                    subplot_size=3):
    
    """
    Plot differences multi-scale spectrograms of original audio x, and resynth audio y, given num losses (scales), min fft size and a subplot size  

    Parameters
    ----------
    x : np.array or torch tensor (samples,)
        original audio data to plot
    y : np.array or torch tensor (samples,)
        resynth audio data to plot
    standard_scale : bool
        if True, all subplots are given the same scale, based on global max and min
    num_losses : int
        number of spectrograms scales to compute
    min_fft_size : int
        size of smallest fft, each size after doubles
    subplot_size : int, float
        size of each subplot
    
    """

    # store diff arrays
    diff_stfts = []
    diff_log_stfts = []

    # # get window sizes
    fft_sizes = [min_fft_size * (2**i) for i in range(num_losses)]

    # set up plot
    figsize = (subplot_size*num_losses, subplot_size*2)
    fig, ax = plt.subplots(2,num_losses, figsize=figsize)

    # compute stft and log_stft for x
    x = numpy_to_torch(x)
    y = numpy_to_torch(y)

    # for each fft size
    for i, fft_size in enumerate(fft_sizes):
        hop_length = int(fft_size / 4)  # overlap by 75%

        # x 
        x_stft = stft(x, 
                    n_fft=fft_size, 
                    hop_length=hop_length, 
                    win_length=fft_size,
                    window=torch.hann_window(fft_size, device=x.device), #move window to gpu
                    normalized=True,
                    return_complex=True).abs()
        x_log_stft = safe_log(x_stft)
        x_stft = torch_to_numpy(x_stft)
        x_log_stft = torch_to_numpy(x_log_stft)

        # y
        y_stft = stft(y, 
                    n_fft=fft_size, 
                    hop_length=hop_length, 
                    win_length=fft_size,
                    window=torch.hann_window(fft_size, device=y.device), #move window to gpu
                    normalized=True,
                    return_complex=True).abs()
        y_log_stft = safe_log(y_stft)
        y_stft = torch_to_numpy(y_stft)
        y_log_stft = torch_to_numpy(y_log_stft)

        # get diff stfts
        diff_stfts.append(x_stft - y_stft)
        diff_log_stfts.append(x_log_stft - y_log_stft)

    # get vmin and vmax
    vmin = 0
    vmax = 0
    for i, fft_size in enumerate(fft_sizes):
        diff_stft = diff_stfts[i]
        if np.amin(diff_stft) < vmin:
            vmin = np.amin(diff_stft)
        if np.amax(diff_stft) > vmax:
            vmax = np.amax(diff_stft)

        diff_log_stft = diff_log_stfts[i]
        if np.amin(diff_log_stft) < vmin:
            vmin = np.amin(diff_log_stft)
        if np.amax(diff_log_stft) > vmax:
            vmax = np.amax(diff_log_stft)

    # use the positive of the max bound for vmax and -bound for vmin
    bound = max(vmax, abs(vmin))
    vmax = bound
    vmin = -1 * bound

    # plot each fft size
    for i, fft_size in enumerate(fft_sizes):

        diff_stft = diff_stfts[i]
        diff_log_stft = diff_log_stfts[i]

        # Make plots
        ax[0, i].set_title(f"FFT size: {fft_size}")
        if standard_scale:
            ax[0, i].imshow(diff_stft, aspect='auto', origin='lower', interpolation=None, cmap="bwr", vmin=vmin, vmax=vmax)
            ax[1, i].imshow(diff_log_stft, aspect='auto', origin='lower', interpolation=None, cmap="bwr", vmin=vmin, vmax=vmax)
        else:
            ax[0, i].imshow(diff_stft, aspect='auto', origin='lower', interpolation=None, cmap="bwr")
            ax[1, i].imshow(diff_log_stft, aspect='auto', origin='lower', interpolation=None, cmap="bwr")

    # Overall plot settings 
    ax[0, 0].set_ylabel("STFT", rotation=0, size='large')
    ax[1, 0].set_ylabel("Log STFT", rotation=0, size='large')
    ax[0,0].yaxis.set_label_coords(-0.4, 0.4)
    ax[1,0].yaxis.set_label_coords(-0.4, 0.4)
    fig.suptitle(f'Difference of Multi-Scale Spectrograms \nRed=Original > Resynth\nBlue=Resynth > Original', fontsize=16)
    fig.tight_layout()

def plot_harmonic_params(p, idx=0):
    """
    Takes a dictionary of DDSP harmonic oscillator params and makes a plot summarizing them
    
    y-axis is harmonic number, x-axis is frames

    Parameters
    ----------
    p : dict
        p['f0'] : f0 envelope tensor (batch, frames)
        p['overall_amplitude'] : overall amplitude (batch, frames)
        p['harm_distr'] : harmonic distribution (sums to 1) tensor (batch, # harmonics, frames)

    idx : int
        index in batch to plot example from

    Returns
    ----------
    item_dict : dict of np arrays
        Dict of the chosen items' audio, f0, and loudness
    """

    harm_distr = torch_to_numpy(p['harm_distr'][idx]) # (# harmonics, frames)
    f0 = torch_to_numpy(p['f0'][idx]) # (frames)
    overall_amplitude = torch_to_numpy(p['overall_amplitude'][idx]) # (# frames)

    num_osc = harm_distr.shape[0]

    harmonic_mults = np.arange(1, num_osc + 1)

    harm_freqs = f0[None, :] * harmonic_mults[:, None] # (# harmonics, frames)

    nyquist_zero_mask = harm_freqs < 8000.0 # anything below 8000.0 is 1, else 0, shape: (# harmonics, frames)

    hd = harm_distr * overall_amplitude

    hd = hd * nyquist_zero_mask

    plt.figure(figsize = (8,6))
    plt.imshow(hd, aspect='auto', interpolation='none', origin='lower')
    plt.ylabel("Harmonic Number")
    plt.xlabel("Time (s)")
    plt.title("Harmonic Distribution")
    plt.text(0, -20, "Columns are scaled by overall_amplitude. \nFrequencies above Nyquist are zeroed. \nAlso zeroed where f0 == 0.")
    plt.xticks(ticks=[0,(ITEM_DUR*FRAME_RATE)//2, (ITEM_DUR*FRAME_RATE)], labels=["0",f"{ITEM_DUR/2}", f"{ITEM_DUR}"])


