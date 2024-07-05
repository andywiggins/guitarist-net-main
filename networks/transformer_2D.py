# Author: Andy Wiggins <awiggins@drexel.edu>
# Torch nn network for a transformer along two dims
import torch.nn as nn
from globals import *

class Transformer2D(nn.Module):
    """
    Torch nn module for a basic model 2 linear layers around 2 transformers. 
    First along features, second along time dimension.
    Ends with a single linear layer (features)
    """
    def __init__(self,
                num_inputs,
                num_frames=int(ITEM_DUR * FRAME_RATE),
                linear_1_size=TRANSFORMER2D_LINEAR_1_SIZE,
                transformer_dim=TRANSFORMER2D_DIM,
                feat_num_heads=TRANSFORMER2D_FEAT_NUM_HEADS,
                time_num_heads=TRANSFORMER2D_TIME_NUM_HEADS,
                num_layers=TRANSFORMER2D_NUM_LAYERS,
                num_outputs=TRANSFORMER2D_LINEAR_2_SIZE):
        """
        Initialize DDSP decoder module.


        Parameters
        ----------
        num_inputs : int
            size of input features
        linear_1_size : int
            size of 1st dense layer, and input/output of transformer
        transformer_dim : int
            dimension of the transformer model
        num_heads : int
            number of attention heads in the transformer
        num_layers : int
            number of transformer layers
        linear_2_size : int
            size of 2nd dense layer, and number of outputs
        """
        super().__init__()

        self.num_inputs = num_inputs # num features
        self.num_frames = num_frames # num frames
        self.num_outputs = num_outputs

        # create layers
        self.linear1 = nn.Linear(num_inputs, linear_1_size)
        self.leakyReLU1 = nn.LeakyReLU()

        # transformer - attention along features
        feat_transformer_layer = nn.TransformerEncoderLayer(
            d_model=num_frames,
            nhead=feat_num_heads,
            dim_feedforward=transformer_dim,
            batch_first=True
        )
        self.feat_transformer = nn.TransformerEncoder(feat_transformer_layer, num_layers=num_layers)

        # transformer - attention along frames
        time_transformer_layer = nn.TransformerEncoderLayer(
            d_model=linear_1_size,
            nhead=time_num_heads,
            dim_feedforward=transformer_dim,
            batch_first=True
        )
        self.time_transformer = nn.TransformerEncoder(time_transformer_layer, num_layers=num_layers)

        self.linear2 = nn.Linear(linear_1_size, num_outputs)
        self.leakyReLU2 = nn.LeakyReLU()
        self.layer_norm = nn.LayerNorm(num_outputs)

    def forward(self, inputs):
        """
        Process string conditioning and idx outputs synth params.

        Parameters
        ----------
        inputs : tensor (batch, frames, input_size)
            input data to be fed thru model
  
        Returns
        ----------
        outputs : tensor (batch, frames, output_size)
            output data from model
        """

        # feed inputs through
        linear1_out = self.linear1(inputs)
        leaky1_out = self.leakyReLU1(linear1_out) # (batch, frames, features)

        # permute to swap frames and features
        feat_transformer_in = leaky1_out.permute(0,2,1) # (batch, features, frames)

        # apply feature-axis transformer
        feat_transformer_out = self.feat_transformer(feat_transformer_in) # (batch, features, frames)

        # permute to swap frames and features
        time_transformer_in = feat_transformer_out.permute(0,2,1) # (batch, frames, features)

        # apply feature-axis transformer
        time_transformer_out = self.time_transformer(time_transformer_in) # (batch, frames, features)

        linear2_out = self.linear2(time_transformer_out)
        leaky2_out = self.leakyReLU2(linear2_out)
        outputs = self.layer_norm(leaky2_out)

        return outputs

