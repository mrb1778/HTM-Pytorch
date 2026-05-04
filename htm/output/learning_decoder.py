import torch
from torch import nn as nn


class LearningDecoder(nn.Module):
    def __init__(
            self,
            num_cells: int,
            num_inputs: int,
            device: torch.device | str | None = None
    ) -> None:
        super().__init__()
        self.device = device
        self.classifier = nn.Linear(num_cells, num_inputs, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x.float())
