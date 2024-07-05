# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch nn network for a basic cnn model (based on TabCNN)

import torch.nn as nn
from globals import *

class BasicCNN(nn.Module):
    """
    Torch nn module for a basic model containing 2 linear layers around a gated-recurrent unit.
    """
    def __init__(self,
                num_inputs,
                linear_1_size=BASIC_GRU_LINEAR_1_SIZE,
                gru_num_units=BASIC_GRU_GRU_NUM_UNITS,
                num_outputs=BASIC_GRU_LINEAR_2_SIZE):
        """
        Initialize DDSP decoder module.


        Parameters
        ----------
        num_inputs : int
            size of input features
        linear_1_size : int
            size of 1st dense layer
        gru_num_units : int
            unit size of gru
        linear_2_size : int
            size of 2nd dense layer, and number of outputs
        """
        super().__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # create layers
        self.conv1 = nn.Conv2d(16, 33, (3, 5), padding=(4, 2))
        self.conv2 = nn.Conv2d(16, 33, (3, 5), padding=(4, 2))
        self.conv3 = nn.Conv2d(16, 33, (3, 5), padding=(4, 2))
        self.leakyReLU1 = nn.LeakyReLU()
        self.gru = nn.GRU(input_size=linear_1_size,
                        hidden_size=gru_num_units,
                        batch_first=True)
        self.linear2 = nn.Linear(gru_num_units, num_outputs)
        self.leakyReLU2 = nn.LeakyReLU()
        self.layer_norm = nn.LayerNorm(num_outputs)


    def forward(self, inputs):
        """
        Process string conditioning and idx outputs synth params.

        Parameters
        ----------
        inputs : tensor (batch, frames, feature_size)
            input data to be fed thru model
  
        Returns
        ----------
        outputs : tensor (batch, frames, output_size)
            output data from model
        """

        # feed inputs thru
        linear1_out = self.linear1(inputs)
        leaky1_out = self.leakyReLU1(linear1_out)
        gru_out = self.gru(leaky1_out)[0]
        linear2_out = self.linear2(gru_out)
        leaky2_out = self.leakyReLU2(linear2_out)
        outputs = self.layer_norm(leaky2_out)

        return outputs