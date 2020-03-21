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
                    .repeat(s_emb.size(0), 1)
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
    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        self._init_configuration(config, configuration_key)

        # This is per default reciprocal
        # Initialize base model
        # Using a dataset with twice the number of relations to initialize base model
        alt_dataset = dataset.shallow_copy()
        alt_dataset._num_relations = dataset.num_relations() * 2
        base_model = KgeModel.create(
            config, alt_dataset, self.configuration_key + ".base_model"
        )

        # Initialize this model
        super().__init__(
            config, dataset, base_model.get_scorer(), initialize_embedders=False
        )
        self._base_model = base_model

        self._entity_embedder = self._base_model.get_s_embedder()
        self._relation_embedder = self._base_model.get_p_embedder()

    def prepare_job(self, job, **kwargs):
        self._base_model.prepare_job(job, **kwargs)

    def penalty(self, **kwargs):
        return super().penalty(**kwargs) + self._base_model.penalty(**kwargs)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:

        if direction == "o":

            o_all_emb = self.get_o_embedder().embed_all()
            o_pr = torch.logsumexp(o_all_emb[o], dim=1) - torch.logsumexp(
                o_all_emb, dim=(0, 1)
            )

            s_all_emb = self.get_s_embedder().embed_all()
            p_all_emb = self.get_p_embedder().embed_all()[
                : self.dataset.num_relations()
            ]
            po_pr = self._scorer.score_emb(
                torch.ones_like(s_all_emb[s]), p_all_emb, o_all_emb[o], combine="s*o"
            )[torch.arange(len(o)), p] - torch.logsumexp(
                self._scorer.score_emb(
                    torch.ones_like(s_all_emb[s]),
                    p_all_emb,
                    o_all_emb[o],
                    combine="s*o",
                ),
                dim=1,
            )

            spo_pr = self._scorer.score_emb(
                s_all_emb, p_all_emb[p], o_all_emb[o], combine="*po"
            )[torch.arange(len(o)), s] - torch.logsumexp(
                self._scorer.score_emb(
                    s_all_emb, p_all_emb[p], o_all_emb[o], combine="*po"
                ),
                dim=1,
            )

            # joint log probability  log p(s,p,o)
            return o_pr + po_pr + spo_pr
        # use recirocal relations
        elif direction == "s":
            s_all_emb = self.get_s_embedder().embed_all()
            s_pr = torch.logsumexp(s_all_emb[s], dim=1) - torch.logsumexp(
                s_all_emb, dim=(0, 1)
            )

            o_all_emb = self.get_o_embedder().embed_all()
            p_all_emb = self.get_p_embedder().embed_all()[
                self.dataset.num_relations() :
            ]
            ps_pr = self._scorer.score_emb(
                torch.ones_like(o_all_emb[o]), p_all_emb, s_all_emb[s], combine="s*o"
            )[torch.arange(len(s)), p] - torch.logsumexp(
                self._scorer.score_emb(
                    torch.ones_like(o_all_emb[o]),
                    p_all_emb,
                    s_all_emb[s],
                    combine="s*o",
                ),
                dim=1,
            )

            spo_pr = self._scorer.score_emb(
                o_all_emb, p_all_emb[p], s_all_emb[s], combine="*po"
            )[torch.arange(len(s)), o] - torch.logsumexp(
                self._scorer.score_emb(
                    o_all_emb, p_all_emb[p], s_all_emb[s], combine="*po"
                ),
                dim=1,
            )

            # joint log probability with reciprocal relation
            return s_pr + ps_pr + spo_pr
        else:
            raise Exception(
                "Reciprocal models cannot compute " "undirected spo scores."
            )

    def score_sp(self, s, p, o=None):
        raise NotImplementedError
        # if o == None:
        #     o = torch.arange(self.dataset.num_entities())
        #
        # o_all_emb = self.get_o_embedder().embed_all()
        # o_pr = torch.exp(o_all_emb[o]).sum(axis=1) / torch.exp(o_all_emb).sum()
        #
        # s_all_emb = self.get_s_embedder().embed_all()
        # p_all_emb = self.get_p_embedder().embed_all()[:self.dataset.num_relations()]
        # po_norm = self._scorer.score_emb(torch.ones_like(o_all_emb[o]), p_all_emb,
        #                                  o_all_emb[o], combine="s*o")
        # po_pr = torch.exp(po_norm[:,p]).transpose(0,1) / torch.exp(po_norm).sum(axis=1)
        #
        # spo_norm = self._scorer.score_emb(s_all_emb, p_all_emb[p], o_all_emb[o],
        #                                   combine="*po")
        # spo_pr = spo_norm[torch.arange(len(o)), s] / spo_norm.sum(axis=1)
        #
        # # joint probability p(s,p,o)
        # return o_pr * po_pr * spo_pr

    def score_po(self, p, o, s=None):
        raise NotImplementedError

    def score_so(self, s, o, p=None):
        raise Exception("This is a reciprocal model and cannot score relations.")

    def score_sp_po(
        self,
        s: torch.Tensor,
        p: torch.Tensor,
        o: torch.Tensor,
        entity_subset: torch.Tensor = None,
    ) -> torch.Tensor:

        n = len(entity_subset)
        scores_sp = (
            self.score_spo(
                s.repeat(n), p.repeat(n), entity_subset.repeat(len(s)), direction="o"
            )
            .view(-1, len(s))
            .transpose(0, 1)
        )

        scores_po = (
            self.score_spo(
                s.repeat(n), p.repeat(n), entity_subset.repeat(len(s)), direction="s"
            )
            .view(-1, len(s))
            .transpose(0, 1)
        )

        return torch.cat((scores_sp, scores_po), dim=1)
