import torch
from torch import Tensor
from kge import Config, Dataset
from kge.model.kge_model import KgeModel

import math
import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel

class GenerativeScorer(RelationalScorer):
    """Generative scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self._base_scorer: RelationalScorer

    def score_emb(self, s_emb: Tensor, p_emb: Tensor, o_emb: Tensor, combine: str):
        if combine == "spo":
            return torch.log(torch.exp(s_emb).sum(axis=1)).view(-1, 1) +\
                self._base_scorer.score_emb(s_emb, p_emb, torch.ones_like(s_emb), combine) +\
                self._base_scorer.score_emb(s_emb, p_emb, o_emb, combine)
        elif combine == "sp*":
            return torch.log(torch.exp(s_emb).sum(axis=1)).view(-1, 1).repeat(1,o_emb.size(0)) + \
                   self._base_scorer.score_emb(s_emb, p_emb, torch.ones_like(o_emb), combine) +\
                   self._base_scorer.score_emb(s_emb, p_emb, o_emb, combine)
        elif combine == "*po":
            return torch.log(torch.exp(s_emb).sum(axis=1)).view(1, -1).repeat(o_emb.size(0),1) + \
                   self._base_scorer.score_emb(s_emb, p_emb, torch.ones_like(o_emb), combine) + \
                   self._base_scorer.score_emb(s_emb, p_emb, o_emb, combine)


class GenerativeModel(KgeModel):
    """Generative model"""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        self._init_configuration(config, configuration_key)

        # Initialize base model
        base_model = KgeModel.create(
            config, dataset, self.configuration_key + ".base_model"
        )

        # Initialize this model
        super().__init__(
            config, dataset, GenerativeScorer, initialize_embedders=False
        )
        self._base_model = base_model
        self.get_scorer()._base_scorer = self._base_model.get_scorer()
        # TODO change entity_embedder assignment to sub and obj embedders when support
        # for that is added
        self._entity_embedder = self._base_model.get_s_embedder()
        self._relation_embedder = self._base_model.get_p_embedder()

    def prepare_job(self, job, **kwargs):
        self._base_model.prepare_job(job, **kwargs)

    def penalty(self, **kwargs):
        return super().penalty(**kwargs) + self._base_model.penalty(**kwargs)





