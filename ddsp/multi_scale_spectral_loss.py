# Author: Andy Wiggins <awiggins@drexel.edu>
# Function for multi-scale spectral loss

from torch import stft
import torch.nn as nn
import numpy as np
from util import safe_log
from globals import *


def multi_scale_spectral_loss(x, y, 
                            num_losses=NUM_SPECTRAL_LOSSES, 
                            min_fft_size=MIN_SPECTRAL_LOSS_FFT_SIZE,
                            log_weighting=SPECTRAL_LOSS_LOG_WEIGHTING,
                            linear_weighting=SPECTRAL_LOSS_LINEAR_WEIGHTING,
                            normalize_audio=NORMALIZE_AUDIO_IN_LOSS_FUNCTION,
                            normalize_in_stft=NORMALIZE_IN_LOSS_STFT):
    """
    Computes multi-scale spectral loss between x and y

    Parameters
    ----------
    x : torch tensor (batch, samples)...can also be just (samples)
        tensor to compare loss function with (audio)
    y : torch tensor (batch, samples)...can also be just (samples)
        tensor to compare loss function with (audio)
    num_losses : int
        number of spectral losses to compute
    min_win_size : int
        starting window size, doubles each time
    weighting : float
        amount to weight log spectrogram comparison
    normalize_audio : bool
        if True, audios x and y will be normalized (p=np.inf) before comparing
        (inf normalization scales the signal to [-1,1])

    Returns
    ----------
    loss : tensor
        loss tensor
    """

    # normalize audios x and y
    # normalizes along dim=1
    if normalize_audio:
        x = nn.functional.normalize(x, p=np.inf, dim=1)
        y = nn.functional.normalize(y, p=np.inf, dim=1)

    # initialize loss to zero
    loss = 0

    # get window sizes
    fft_sizes = [min_fft_size * (2**i) for i in range(num_losses)]

    # for each fft size
    for fft_size in fft_sizes:
        hop_length = int(fft_size / 4)  # overlap by 75%
        
        # compute stfts for x
        x_stft = stft(x, 
                    n_fft=fft_size, 
                    hop_length=hop_length, 
                    win_length=fft_size,
                    window=torch.hann_window(fft_size, device=x.device), #move window to gpu
                    normalized=normalize_in_stft,
                    return_complex=True).abs()

        # compute stfts for y
        y_stft = stft(y, 
                    n_fft=fft_size, 
                    hop_length=hop_length, 
                    win_length=fft_size,
                    window=torch.hann_window(fft_size, device=y.device), #move window to gpu
                    normalized=normalize_in_stft,
                    return_complex=True).abs()

        # determine the cut off bin to optionally boost treble
        num_freq_bins = x_stft.shape[0] # 0th axis is freq bins
        cut_off_bin = int(num_freq_bins * LOSS_FUNCTION_TREBLE_BOOST_FRAC)

        # compute linear and log losses
        ## linear loss
        linear_abs_diff = (x_stft - y_stft).abs()
        # boost treble in stft, by amt above cut off bin
        linear_abs_diff[0:cut_off_bin, :] *= LOSS_FUNCTION_TREBLE_BOOST_AMT
        linear_loss = linear_abs_diff.mean() # taking the mean here means the loss is being scaled by batch size
        
        ## log loss
        log_abs_diff = (safe_log(x_stft) - safe_log(y_stft)).abs()
        log_abs_diff[0:cut_off_bin, :] *= LOSS_FUNCTION_TREBLE_BOOST_AMT
        log_loss = log_abs_diff.mean()

        # weighted sum of linear and log losses
        L_i = linear_weighting * linear_loss + log_weighting * log_loss

        # add on to total loss
        loss += L_i

    return loss
    
            




