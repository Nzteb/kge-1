import torch
from torch import Tensor
import kge.model
from kge import Config, Dataset
from kge.model.kge_model import KgeModel

import math
import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from pydoc import locate


class GenerativeScorer(RelationalScorer):
    """Generative scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        _scorer_type = self.get_option("base_scorer")
        self._base_scorer: RelationalScorer = locate("kge.model." + _scorer_type)(
            Config, Dataset, configuration_key=None
        )

    def score_emb(self, s_emb: Tensor, p_emb: Tensor, o_emb: Tensor, combine: str):

        # score p(s,p,o) = p(o)p(p|o)p(s|p,o)
        if True:
            if combine == "spo":
                return (
                    torch.log(torch.exp(o_emb).sum(axis=1)).view(-1, 1)
                    + self._base_scorer.score_emb(
                        torch.ones_like(s_emb), p_emb, o_emb, combine
                    )
                    + self._base_scorer.score_emb(s_emb, p_emb, o_emb, combine)
                )
            elif combine == "sp*":
                return (
                    torch.log(torch.exp(o_emb).sum(axis=1))
                    .view(1, -1)
                    .repeat(s_emb.size(0),1)
                    + self._base_scorer.score_emb(
                        torch.ones_like(s_emb), p_emb, o_emb, combine
                    )
                    + self._base_scorer.score_emb(s_emb, p_emb, o_emb, combine)
                )
            elif combine == "*po":
                raise Exception(
                    "Can only rank heads, i.e., be used with reciprocal model"
                )


class GenerativeModel(KgeModel):
    """Generative model"""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(
            config, dataset, GenerativeScorer, configuration_key=configuration_key
        )
    def change_mode(self, mode:str):
        if mode == "valid":
            self.get_scorer().mode = "valid"
        elif mode == "train":
            self.get_scorer().mode = "train"
        else:
            raise Exception("Wrong scorer mode for generative scorer")

