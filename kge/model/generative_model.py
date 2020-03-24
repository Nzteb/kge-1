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
        self.mode = "score"
        self._entity_embedder = self._base_model.get_s_embedder()
        self._relation_embedder = self._base_model.get_p_embedder()

    def prepare_job(self, job, **kwargs):
        self._base_model.prepare_job(job, **kwargs)

    def penalty(self, **kwargs):
        return super().penalty(**kwargs) + self._base_model.penalty(**kwargs)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        # learn the joint distribution in training
        # p(s,p,o) = p(s)p(p|s)p(o|s,p); and also for reciprocal facts
        if self.mode == "train":
            return self._score_spo(s, p, o, direction=direction)
        # score with the conditional which produces the same ranks as the
        # raw scoring function
        elif self.mode == "score":
            if direction == "o":
                return super().score_spo(s, p, o, "o")
            elif direction == "s":
                return super().score_spo(o, p + self.dataset.num_relations(), s, "o")
            else:
                raise Exception("Cannot compute " "undirected spo scores.")
        else:
            raise ValueError("Wrong mode for generative model.")

    def _score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:

        if direction == "o":
            s_all_emb = self.get_s_embedder().embed_all()
            s_pr = torch.logsumexp(s_all_emb[s], dim=1) - torch.logsumexp(
                s_all_emb, dim=(0, 1)
            )

            o_all_emb = self.get_o_embedder().embed_all()
            p_all_emb = self.get_p_embedder().embed_all()[
                : self.dataset.num_relations()
            ]

            ps_pr = self._scorer.score_emb(
                s_all_emb[s], p_all_emb, torch.ones_like(o_all_emb[o]), combine="s*o"
            )
            ps_pr = ps_pr[torch.arange(len(s)), p] - torch.logsumexp(ps_pr, dim=1)
            # TODO: you might want to do the scoring only once..
            spo_pr = self._scorer.score_emb(
                s_all_emb[s], p_all_emb[p], o_all_emb, combine="sp*"
            )
            spo_pr = spo_pr[torch.arange(len(s)), o] - torch.logsumexp(spo_pr, dim=1)

            # joint log probability
            return s_pr + ps_pr + spo_pr

        elif direction == "s":

            o_all_emb = self.get_o_embedder().embed_all()
            o_pr = torch.logsumexp(o_all_emb[o], dim=1) - torch.logsumexp(
                o_all_emb, dim=(0, 1)
            )

            # TODO reciprocal yes no
            s_all_emb = self.get_s_embedder().embed_all()
            p_all_emb = self.get_p_embedder().embed_all()[
                self.dataset.num_relations() :
            ]
            po_pr = self._scorer.score_emb(
                o_all_emb[o], p_all_emb, torch.ones_like(s_all_emb[s]), combine="s*o"
            )
            po_pr = po_pr[torch.arange(len(o)), p] - torch.logsumexp(po_pr, dim=1)

            spo_pr = self._scorer.score_emb(
                o_all_emb[o], p_all_emb[p], s_all_emb, combine="sp*"
            )

            spo_pr = spo_pr[torch.arange(len(o)), s] - torch.logsumexp(spo_pr, dim=1)

            # joint log probability  log p(s,p,o)
            return o_pr + po_pr + spo_pr
        # use recirocal relations

        else:
            raise Exception(
                "Reciprocal models cannot compute " "undirected spo scores."
            )

    def score_sp(self, s, p, o=None):
        raise NotImplementedError

    def score_po(self, p, o, s=None):
        raise NotImplementedError

    def score_so(self, s, o, p=None):
        raise Exception(
            "This is a reciprocal model and cannot score undirected relations."
        )

    def score_sp_po(
        self,
        s: torch.Tensor,
        p: torch.Tensor,
        o: torch.Tensor,
        entity_subset: torch.Tensor = None,
    ) -> torch.Tensor:

        n = len(entity_subset)
        m = len(s)

        triples = torch.cat(
            (
                torch.cat((s.view(-1, 1), p.view(-1, 1)), dim=1)
                .repeat(1, n)
                .view(-1, 2),
                entity_subset.view(-1, 1).repeat(m, 1),
            ),
            dim=1,
        )
        scores_sp = self.score_spo(
            triples[:, 0], triples[:, 1], triples[:, 2], direction="o"
        ).view(len(s), -1)

        triples = torch.cat(
            (
                entity_subset.view(-1, 1).repeat(m, 1),
                torch.cat((p.view(-1, 1), o.view(-1, 1)), dim=1)
                .repeat(1, n)
                .view(-1, 2),
            ),
            dim=1,
        )

        scores_po = self.score_spo(
            triples[:, 0], triples[:, 1], triples[:, 2], direction="s"
        ).view(len(s), -1)

        return torch.cat((scores_sp, scores_po), dim=1)
