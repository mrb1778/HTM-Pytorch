from typing import List, Any

import torch
from torch import nn as nn

from xutils.dl.pytorch import utils as pyu


class IterableSparseEncoder(nn.Module):
    """
    A simple encoder to convert iterable items into Sparse Distributed Representations (SDRs).

    This encoder uses a hash-based approach to generate a unique, fixed-sparsity SDR
    for each item. This is suitable for a small, categorical input space.
    """

    def __init__(
            self,
            items: List[Any],
            output_size: int,
            sparsity: float = 0.02,
            device: torch.device | str | None = None
    ) -> None:
        """
        Initializes the encoder with SDR parameters.

        Args:
            output_size (int): The total number of bits in the SDR.
            sparsity (int): The number of active bits (the sparsity).
            seed (int): The base seed for hash generation to ensure reproducibility.
            numeric_encoder (Callable[[Any], int]): A function to convert items to numeric codes, required if not numeric.
        """
        super(IterableSparseEncoder, self).__init__()

        assert 0 < sparsity <= output_size, "sparsity must be between 1 and output_size."

        self.output_size = output_size
        self.sparsity = int(self.output_size * sparsity)
        self.device = device

        self.items = items
        self.register_buffer(
            "encodings",
            pyu.create_sparse_tensor(rows=len(items),
                                     cols=self.output_size,
                                     active_per_row=self.sparsity,
                                     device=self.device))

    def forward(self, x: Any) -> torch.Tensor:
        """
    Encodes a single item into its corresponding SDR.

    Args:
        x (Any): The item to encode. Must be in the list of items passed to the constructor.

    Returns:
        torch.Tensor: The SDR for the character.
    """
        # if len(x) != 1:
        #     raise ValueError("Input must be a single character string.")

        if x not in self.items:
            raise ValueError(f"input {x} not in {self.items}")

        return self.encodings[self.items.index(x)]
