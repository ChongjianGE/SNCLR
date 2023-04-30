import torch
from .memory_bank import MemoryBankModule


class NNMemoryBankModuleNormMultiKeep(MemoryBankModule):
    """Nearest Neighbour Memory Bank implementation

    This class implements a nearest neighbour memory bank as described in the
    NNCLR paper[0]. During the forward pass we return the nearest neighbour
    from the memory bank.

    [0] NNCLR, 2021, https://arxiv.org/abs/2104.14548

    Attributes:
        size:
            Number of keys the memory bank can store. If set to 0,
            memory bank is not used.

    Examples:
        >>> model = NNCLR(backbone)
        >>> criterion = NTXentLoss(temperature=0.1)
        >>>
        >>> nn_replacer = NNmemoryBankModule(size=2 ** 16)
        >>>
        >>> # forward pass
        >>> (z0, p0), (z1, p1) = model(x0, x1)
        >>> z0 = nn_replacer(z0.detach(), update=False)
        >>> z1 = nn_replacer(z1.detach(), update=True)
        >>>
        >>> loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
    """
    def __init__(self, size: int = 2 ** 16, topk=5):
        super(NNMemoryBankModuleNormMultiKeep, self).__init__(size)
        self.topk = topk

    def forward(self,
                output: torch.Tensor,
                update: bool = False):
        """Returns nearest neighbour of output tensor from memory bank

        Args:
            output: The torch tensor for which you want the nearest neighbour
            update: If `True` updated the memory bank by adding output to it

        """

        output, bank = \
            super(NNMemoryBankModuleNormMultiKeep, self).forward(output, update=update)
        bank = bank.to(output.device).t()

        output_normed = torch.nn.functional.normalize(output, dim=1)
        bank_normed = torch.nn.functional.normalize(bank, dim=1)

        similarity_matrix = \
            torch.einsum("nd,md->nm", output_normed, bank_normed)

        index_nearest_neighbours = torch.topk(similarity_matrix, self.topk)[1].reshape(-1)

        nearest_neighbours = torch.index_select(bank, dim=0, index=index_nearest_neighbours)
        b, dim = output.shape

        res = torch.cat((output.unsqueeze(dim=1), nearest_neighbours.reshape(b, self.topk, dim)), dim=1).reshape(-1, dim)
        return nearest_neighbours, res
