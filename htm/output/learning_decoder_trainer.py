from typing import Callable, Any, List, Optional

import torch
from torch import nn as nn, optim

from .learning_decoder import LearningDecoder


class LearningDecoderTrainer:
    def __init__(
            self,
            model: LearningDecoder,
            numeric_encoder: Callable[[Any], int] = None,
            numeric_decoder: Callable[[int], Any] = None,
            device: torch.device | str | None = None
    ) -> None:
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.device = device
        self.numeric_encoder = numeric_encoder
        self.numeric_decoder = numeric_decoder
        self.learn = True

    def train(self, inputs, value, numeric_value=None) -> tuple[torch.Tensor | None, bool | None]:
        if self.learn:
            self.model.train()
            self.optimizer.zero_grad()

            inputs = inputs.reshape([1, -1])
            output = self.model(inputs)
            if numeric_value is None:
                numeric_value = self.numeric_encoder(value) if self.numeric_encoder is not None else value
            label = torch.tensor([numeric_value], dtype=torch.long, device=self.device)

            prediction = torch.argmax(output, dim=1).item()
            is_correct = (prediction == numeric_value)

            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            return loss.item(), is_correct
        else:
            return None, None

    def decode(self, x: torch.Tensor) -> Any:
        self.model.eval()
        with torch.no_grad():
            x = x.reshape([1, -1])
            output = self.model(x)
            predicted = torch.argmax(output, dim=1).item()
            return self.numeric_decoder(predicted) if self.numeric_decoder is not None else predicted
