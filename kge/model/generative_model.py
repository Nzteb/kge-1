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
        if combine == "spo":
            return (
                torch.log(torch.exp(s_emb).sum(axis=1)).view(-1, 1)
                + self._base_scorer.score_emb(
                    s_emb, p_emb, torch.ones_like(s_emb), combine
                )
                + self._base_scorer.score_emb(s_emb, p_emb, o_emb, combine)
            )
        elif combine == "sp*":
            return (
                torch.log(torch.exp(s_emb).sum(axis=1))
                .view(-1, 1)
                .repeat(1, o_emb.size(0))
                + self._base_scorer.score_emb(
                    s_emb, p_emb, torch.ones_like(o_emb), combine
                )
                + self._base_scorer.score_emb(s_emb, p_emb, o_emb, combine)
            )
        elif combine == "*po":
            return (
                torch.log(torch.exp(s_emb).sum(axis=1))
                .view(1, -1)
                .repeat(o_emb.size(0), 1)
                + self._base_scorer.score_emb(
                    s_emb, p_emb, torch.ones_like(o_emb), combine
                )
                + self._base_scorer.score_emb(s_emb, p_emb, o_emb, combine)
            )


class GenerativeModel(KgeModel):
    """Generative model"""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(
            config, dataset, GenerativeScorer, configuration_key=configuration_key
        )
