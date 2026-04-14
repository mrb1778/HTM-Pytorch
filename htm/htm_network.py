import torch
from torch import nn as nn

from .output.learning_decoder_trainer import LearningDecoderTrainer
from .spatial_pooler import SpatialPooler


class HtmNetwork(nn.Module):
    def __init__(self,
                 spatial_pooler: SpatialPooler,
                 temporal_memory: nn.Module,
                 encoder: nn.Module = None,
                 decoder: nn.Module = None,
                 decoder_trainer: LearningDecoderTrainer = None,
                 device: str = "cpu"):
        super(HtmNetwork, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_trainer = decoder_trainer
        self.device = device

        self.spatial_pooler = spatial_pooler
        self.temporal_memory = temporal_memory

    def reset(self):
        self.temporal_memory.reset()

    def forward(self,
                x: str = None,
                x_encoded: str = None,
                x_sdr: torch.Tensor = None,
                run_temporal_memory: bool = True) -> str | nn.Module:
        # 1. Encode the input character
        if x_sdr is None:
            if self.encoder is not None:
                assert x_encoded is not None or x is not None, \
                    "either input_encoded or input_text must be provided"
            x_sdr = self.encoder(x_encoded=x_encoded) if x_encoded is not None else self.encoder(x=x)

        # 2. Run the Spatial Pooler
        active_columns = self.spatial_pooler(x_sdr)
        # print(f"Active SP columns: {active_columns.nonzero()}")  #, active_columns)
        if not run_temporal_memory:
            return active_columns
        else:
            # 3. Run the Temporal Memory
            predicted_cells = self.temporal_memory(active_columns)

            if self.decoder_trainer is not None:
                self.decoder_trainer.train(active_columns, x)

            if len(predicted_cells.shape) > 1:
                predicted_cells = predicted_cells.any(dim=1)
            if self.decoder is None and self.decoder_trainer is not None:
                return self.decoder_trainer.decode(predicted_cells)
            else:
                return self.decoder(predicted_cells)
