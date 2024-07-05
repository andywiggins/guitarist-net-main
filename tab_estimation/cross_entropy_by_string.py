# Author: Andy Wiggins <awiggins@drexel.edu>
# Function for cross-entropy loss by string for guitar tab estimation

import torch.nn.functional as F
from globals import *


def cross_entropy_by_string(input, target):
    """
    Computes cross entropy by guitar string between tablatures x and y

    Parameters
    ----------
    intput : torch tensor (batch, frames, 6, fret_classes)
        estimated tablature to compare with loss function
        values in [0,1]
    target : torch tensor (batch, frames, 6, fret_classes)
        ground truth tablature to compare with loss function
        values either 0 or 1

    Returns
    ----------
    loss : tensor
        loss tensor
    """

    # reshape to combine batches + frames
    input = input.flatten(0, 1) # (batch x frames, 6, fret_classes)
    target = target.flatten(0, 1) # (batch x frames, 6, fret_classes)

    # initialize overall loss to zero
    loss = 0

    # get loss for each guitar string
    for string_idx in range(6):

        # slice tab for current string
        input_string_tab = input[:, string_idx, :]  # (batch x frames, fret_classes)
        target_string_tab = target[:, string_idx, :]  # (batch x frames, fret_classes)
        
        # compute string loss via cross entropy
        string_loss = F.cross_entropy(input_string_tab, target_string_tab).mean()  # mean to get a single loss value across batches+frames

        loss += string_loss

    return loss
    
            




