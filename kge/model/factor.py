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
        # Using a dataset with twice the number of relations to initialize base model
        # TODO: you might want to train this reciprocal
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
        # TODO change entity_embedder assignment to sub and obj embedders when support
        # for that is added
        self._entity_embedder = self._base_model.get_s_embedder()
        self._relation_embedder = self._base_model.get_p_embedder()

        # add weights for numeric features per relations
        self.feature_string = self.config.get(self.configuration_key + ".features")
        # num rel x 1 +  2*num features
        # first column the the kge terms; next part subject features then object
        self.num_attributes = self.config.get(
            f"dataset.files.{self.feature_string}.num_attributes"
        )

        weights = torch.zeros(dataset.num_relations(), self.num_attributes*2+1)
        weights[:, 0] = 1
        self._weights = torch.nn.Parameter(weights)


    def prepare_job(self, job, **kwargs):
        self._base_model.prepare_job(job, **kwargs)

    def penalty(self, **kwargs):
        return super().penalty(**kwargs) + self._base_model.penalty(**kwargs)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        o_emb = self.get_o_embedder().embed(o)


        # TODO outfactor the part that greps features s.t. it can be
        #  optimized, reused, or even used in collate

        sparse_values = self.dataset.index(f"entity_numeric_sparse").transpose(0, 1)

        # collect features for subjects
        s_idx = torch.cat((torch.arange(s.size(0)).view(1, -1), s.view(1, -1)))
        s_val = torch.ones(s.size(0)).double()
        s_sparse = torch.sparse.FloatTensor(
            s_idx, s_val, torch.Size((s.size(0), self.dataset.num_entities()))
        ).transpose(0, 1)
        # TODO no need to define s_sparse when it is densified here
        # batch size x num features holding all numeric features for s of every spo
        values_s = (
            torch.sparse.mm(sparse_values, s_sparse.to_dense()).transpose(0, 1).float()
        )

        # collect features for objects
        o_idx = torch.cat((torch.arange(o.size(0)).view(1, -1), o.view(1, -1)))
        o_val = torch.ones(o.size(0)).double()
        o_sparse = torch.sparse.FloatTensor(
            o_idx, o_val, torch.Size((o.size(0), self.dataset.num_entities()))
        ).transpose(0, 1)
        # TODO no need to define o_sparse when it is densified here
        # batch size x num features holding all numeric features for o of every spo
        values_o = (
            torch.sparse.mm(sparse_values, o_sparse.to_dense()).transpose(0, 1).float()
        )
        # scoring
        kge_scores = self.get_scorer().score_emb_spo(s_emb, p_emb, o_emb)

        scores = (
            self._weights[p,] * torch.cat((kge_scores, values_s, values_o), dim=1)
        ).sum(dim=1)
        print("debug")

        return scores

    def score_sp(self, s: Tensor, p: Tensor, o: Tensor = None):
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        if o is None:
            o = torch.arange(self.dataset.num_entities())
            o_emb = self.get_o_embedder().embed_all()
        else:
            o_emb = self.get_o_embedder().embed(o)

        sparse_values = self.dataset.index(f"entity_numeric_sparse").transpose(0, 1)

        # collect features for subjects
        s_idx = torch.cat((torch.arange(s.size(0)).view(1, -1), s.view(1, -1)))
        s_val = torch.ones(s.size(0)).double()
        s_sparse = torch.sparse.FloatTensor(
            s_idx, s_val, torch.Size((s.size(0), self.dataset.num_entities()))
        ).transpose(0, 1)
        # TODO no need to define s_sparse when it is densified here
        # batch size x num features holding all numeric features for s of every spo
        values_s = (
            torch.sparse.mm(sparse_values, s_sparse.to_dense()).transpose(0, 1).float()
        )
        # scoring
        # num_o x batch_size
        kge_scores = self.get_scorer().score_emb(s_emb, p_emb, o_emb, combine="sp_").transpose(0, 1)

        kge_weights = self._weights[p,0].view(-1, p.size(0))

        # broadcasts
        kge_weighted = kge_scores * kge_weights

        sp_weighted = (values_s * self._weights[p, 1:self.num_attributes + 1]).sum(dim=1)

        # num_o x batch_size
        # broadcasts
        sp_kge_weighted = kge_weighted + sp_weighted

        # add the contributions for all the different o's

        # TODO we never have to fully densify the
        #  the full num_ent * num_features vector; considering the mm is optimized
        #  what is left is only num entities x batchsize and batchsize is controllable
        o_weighted = torch.sparse.mm(
            sparse_values.transpose(0,1),
            self._weights[p, self.num_attributes + 1:].transpose(0, 1).double()
        )

        o_weighted = o_weighted[o]

        scores = o_weighted + sp_kge_weighted


        print(self._weights)

        return scores.transpose(0,1)


    def score_po(self, p, o, s=None):
        o_emb = self.get_o_embedder().embed(o)
        p_emb = self.get_p_embedder().embed(p)
        if s is None:
            s = torch.arange(self.dataset.num_entities())
            s_emb = self.get_s_embedder().embed_all()
        else:
            s_emb = self.get_s_embedder().embed(s)

        sparse_values = self.dataset.index(f"entity_numeric_sparse").transpose(0, 1)

        # collect features for objects
        o_idx = torch.cat((torch.arange(o.size(0)).view(1, -1), o.view(1, -1)))
        o_val = torch.ones(o.size(0)).double()
        o_sparse = torch.sparse.FloatTensor(
            o_idx, o_val, torch.Size((o.size(0), self.dataset.num_entities()))
        ).transpose(0, 1)
        # TODO no need to define o_sparse when it is densified here
        # batch size x num features holding all numeric features for o of every spo
        values_o = (
            torch.sparse.mm(sparse_values, o_sparse.to_dense()).transpose(0, 1).float()
        )
        # scoring
        # num_o x batch_size
        kge_scores = self.get_scorer().score_emb(s_emb, p_emb, o_emb,
                                                 combine="_po").transpose(0, 1)

        kge_weights = self._weights[p, 0].view(-1, p.size(0))

        # broadcasts
        kge_weighted = kge_scores * kge_weights

        po_weighted = (values_o * self._weights[p, 1:self.num_attributes + 1]).sum(
            dim=1)

        # num_s x batch_size
        # broadcasts
        po_kge_weighted = kge_weighted + po_weighted

        # add the contributions for all the different o's

        # TODO we never have to fully densify the
        #  the full num_ent * num_features vector; considering the mm is optimized
        #  what is left is only num entities x batchsize and batchsize is controllable
        s_weighted = torch.sparse.mm(
            sparse_values.transpose(0, 1),
            self._weights[p, 1:self.num_attributes + 1].transpose(0, 1).double()
        )

        s_weighted = s_weighted[s]

        scores = s_weighted + po_kge_weighted
        return scores.transpose(0, 1)

    def score_so(self, s, o, p=None):
        raise NotImplementedError

    # def score_sp_po(
    #     self,
    #     s: torch.Tensor,
    #     p: torch.Tensor,
    #     o: torch.Tensor,
    #     entity_subset: torch.Tensor = None,
    # ) -> torch.Tensor:
    #     s = self.get_s_embedder().embed(s)
    #     p_inv = self.get_p_embedder().embed(p + self.dataset.num_relations())
    #     p = self.get_p_embedder().embed(p)
    #     o = self.get_o_embedder().embed(o)
    #     if self.get_s_embedder() is self.get_o_embedder():
    #         if entity_subset is not None:
    #             all_entities = self.get_s_embedder().embed(entity_subset)
    #         else:
    #             all_entities = self.get_s_embedder().embed_all()
    #         sp_scores = self._scorer.score_emb(s, p, all_entities, combine="sp_")
    #         po_scores = self._scorer.score_emb(o, p_inv, all_entities, combine="sp_")
    #     else:
    #         if entity_subset is not None:
    #             all_objects = self.get_o_embedder().embed(entity_subset)
    #             all_subjects = self.get_s_embedder().embed(entity_subset)
    #         else:
    #             all_objects = self.get_o_embedder().embed_all()
    #             all_subjects = self.get_s_embedder().embed_all()
    #         sp_scores = self._scorer.score_emb(s, p, all_objects, combine="sp_")
    #         po_scores = self._scorer.score_emb(o, p_inv, all_subjects, combine="sp_")
    #     return torch.cat((sp_scores, po_scores), dim=1)
