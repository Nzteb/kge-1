import math
import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from pydoc import locate

import torch
from torch import Tensor
from kge import Config, Dataset
from kge.model.kge_model import KgeModel


class FactorGraphModel(KgeModel):
    """Factor Graph Model with support for numerical entity features"""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        self._init_configuration(config, configuration_key)

        # Initialize base model
        base_model = KgeModel.create(
            config, dataset, self.configuration_key + ".base_model"
        )

        # Initialize this model
        super().__init__(
            config,
            dataset,
            base_model.get_scorer(),
            initialize_embedders=False,
            configuration_key=self.configuration_key,
        )
        self._base_model = base_model
        self._entity_embedder = self._base_model.get_s_embedder()
        self._relation_embedder = self._base_model.get_p_embedder()
        self.device = self.config.get("job.device")

        self.feature_file = self.config.get(self.configuration_key + ".feature_file")
        self.num_attributes = self.config.get(
            f"dataset.files.{self.feature_file}.num_attributes"
        )
        self.regularize = self.config.get(
            self.configuration_key + ".regularize_factor_weights"
        )
        self.regularize_p = self.config.get(
            self.configuration_key + ".regularize_factor_weights_args.p"
        )
        self.regularize_weight = self.config.get(
            self.configuration_key + ".regularize_factor_weights_args.regularize_weight"
        )
        # weights for numeric features per relations
        # num rel x 1 + 2*num features
        # first column for the kge scoring; next part subject features then object
        weights = torch.zeros(dataset.num_relations(), self.num_attributes*2+1)
        weights[:, 0] = 1
        self._weights = torch.nn.Parameter(weights)

        # sparse tensor; num_features x num_entities
        self._sparse_values = self.dataset.index(
            f"{self.feature_file}_sparse"
        ).transpose(0, 1).to(self.device)


        self.indicator_only = self.config.get(
            self.configuration_key + ".features_as_indicator"
        )
        if self.indicator_only:
            # Todo you might want to delete the {attribute}_sparse index then
            self._sparse_values = torch.sparse.FloatTensor(
                self._sparse_values._indices(),
                torch.ones_like(self._sparse_values._values()),
                torch.Size(self._sparse_values.size())
            ).to(self.device)

    def prepare_job(self, job, **kwargs):
        self._base_model.prepare_job(job, **kwargs)

    def _factor_weight_penalty(self):
        result = []
        if self.regularize == "" or self.regularize_weight == 0.0:
            pass
        elif self.regularize == "lp":
            p = self.regularize_p
            regularize_weight = self.regularize_weight
            # unweighted Lp regularization
            parameters = self._weights
            if p % 2 == 1:
                parameters = torch.abs(self._weights)
            result += [
                (
                    f"{self.configuration_key}.L{p}_penalty",
                    (regularize_weight / p * (parameters ** p)).sum(),
                )
            ]
        return result

    def penalty(self, **kwargs):
        return super().penalty(**kwargs) + self._base_model.penalty(**kwargs) \
               + self._factor_weight_penalty()

    def get_features(self, e: Tensor):
        """Return a dense feature tensor.

        Returns a len(e) x num_features tensor where tensor[i,j] holds the feature
        value for entity e[i] for feature j. If the value is 0 the entity has no value
        for the corresponding feature.

        """
        # num_entities x len(e) e.g., batch_size

        sparse_coord = torch.zeros(
            (self.dataset.num_entities(), e.size(0))
        ).to(self.device)
        sparse_coord[e, torch.arange(e.size(0))] = 1
        # pick the columns of sparse values (return transpose) corresponding to
        # the entities in e
        return torch.sparse.mm(
            self._sparse_values, sparse_coord
        ).transpose(0, 1)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:

        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        o_emb = self.get_o_embedder().embed(o)

        values_s = self.get_features(s)
        values_o = self.get_features(o)

        # scoring
        kge_scores = self.get_scorer().score_emb_spo(s_emb, p_emb, o_emb)

        scores = (
            self._weights[p, ] * torch.cat((kge_scores, values_s, values_o), dim=1)
        ).sum(dim=1)

        return scores

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None):
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        if o is None:
            o = torch.arange(self.dataset.num_entities())
            o_emb = self.get_o_embedder().embed_all()
        else:
            o_emb = self.get_o_embedder().embed(o)

        values_s = self.get_features(s)
        sp_weighted = (values_s * self._weights[p, 1:self.num_attributes + 1]).sum(
            dim=1)
        # num_o x batch_size
        kge_scores = self.get_scorer().score_emb(
            s_emb,
            p_emb,
            o_emb,
            combine="sp_"
        ).transpose(0, 1)

        kge_weights = self._weights[p, 0].view(-1, p.size(0))
        # broadcasts
        kge_weighted = kge_scores * kge_weights
        # num_o x batch_size; broadcasts
        sp_kge_weighted = kge_weighted + sp_weighted

        # add the contributions for all the different o's
        o_weighted = torch.sparse.mm(
            self._sparse_values.transpose(0,1),
            self._weights[p, self.num_attributes + 1:].transpose(0, 1)
        )
        o_weighted = o_weighted[o]

        scores = o_weighted + sp_kge_weighted
        return scores.transpose(0,1)

    def score_po(self, p, o, s=None):
        o_emb = self.get_o_embedder().embed(o)
        p_emb = self.get_p_embedder().embed(p)
        if s is None:
            s = torch.arange(self.dataset.num_entities())
            s_emb = self.get_s_embedder().embed_all()
        else:
            s_emb = self.get_s_embedder().embed(s)

        values_o = self.get_features(o)
        po_weighted = (values_o * self._weights[p, self.num_attributes + 1:]).sum(
            dim=1)

        # scoring
        # num_o x batch_size
        kge_scores = self.get_scorer().score_emb(
            s_emb,
            p_emb,
            o_emb,
            combine="_po"
        ).transpose(0, 1)
        kge_weights = self._weights[p, 0].view(-1, p.size(0))
        # broadcasts
        kge_weighted = kge_scores * kge_weights

        # num_s x batch_size; broadcasts
        po_kge_weighted = kge_weighted + po_weighted

        # add the contributions for all the different o's
        s_weighted = torch.sparse.mm(
            self._sparse_values.transpose(0,1),
            self._weights[p, 1:self.num_attributes + 1].transpose(0, 1)
        )

        s_weighted = s_weighted[s]
        scores = s_weighted + po_kge_weighted
        return scores.transpose(0, 1)

    def score_so(self, s, o, p=None):
        raise NotImplementedError

    def score_sp_po(
        self,
        s: torch.Tensor,
        p: torch.Tensor,
        o: torch.Tensor,
        entity_subset: torch.Tensor = None,
    ):
        sp_scores = self.score_sp(s, p, entity_subset)
        po_scores = self.score_po(p, o, entity_subset)
        return torch.cat((sp_scores, po_scores), dim=1)
