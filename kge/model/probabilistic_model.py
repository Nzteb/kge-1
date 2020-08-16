import torch
from torch import Tensor
from kge import Config, Dataset
from kge.model.kge_model import KgeModel


class ProbabilisticModel(KgeModel):
    def __init__(
            self,
            config: Config,
            dataset: Dataset,
            configuration_key=None,
            init_for_load_only=False
    ):
        self._init_configuration(config, configuration_key)

        # Initialize base model
        # Using a dataset with twice the number of relations to initialize base model
        alt_dataset = dataset.shallow_copy()
        alt_dataset._num_relations = dataset.num_relations() * 2
        base_model = KgeModel.create(
            config,
            alt_dataset,
            self.configuration_key + ".base_model",
            init_for_load_only=init_for_load_only
        )

        # Initialize this model
        super().__init__(
            config,
            dataset,
            base_model.get_scorer(),
            create_embedders=False,
            init_for_load_only=init_for_load_only
        )
        self._base_model = base_model

        self._entity_embedder = self._base_model.get_s_embedder()
        self._relation_embedder = self._base_model.get_p_embedder()

    def prepare_job(self, job, **kwargs):
        self._base_model.prepare_job(job, **kwargs)

    def penalty(self, **kwargs):
        # TODO penalty computation is moved into probabilistic training
        #  as different training schemes require different penalties aka non-data terms
       return []

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, direction=None) -> Tensor:
        s = self.get_s_embedder().embed(s)
        s_means, s_sigmas = s["means"], s["sigmas"]
        o = self.get_o_embedder().embed(o)
        o_means, o_sigmas = o["means"], o["sigmas"]
        if direction == "o":
            p = self.get_p_embedder().embed(p)
            p_means, p_sigmas = p["means"], p["sigmas"]
            return self._scorer.score_emb(
                s_means,
                p_means,
                o_means,
                combine="spo"
            )
        elif direction == "s":
            p = self.get_p_embedder().embed(p + self.dataset.num_relations())
            p_means, p_sigmas = p["means"], p["sigmas"]
            # reciprocal i.e., flipped triple
            return self._scorer.score_emb(
                o_means,
                p_means,
                s_means,
                combine="spo"
            )
        else:
            raise Exception(
                "Reciprocal models cannot compute "
                "undirected spo scores."
            )

    def score_sp(self, s, p, o=None):
        if o is None:
            o = self.get_o_embedder().embed_all()
        else:
            o = self.get_o_embedder().embed(o)
        o_means, o_sigmas = o["means"], o["sigmas"]
        s = self.get_s_embedder().embed(s)
        s_means, s_sigmas = s["means"], s["sigmas"]
        p = self.get_p_embedder().embed(p)
        p_means, p_sigmas = p["means"], p["sigmas"]
        return self._scorer.score_emb(
            s_means,
            p_means,
            o_means,
            combine="sp*"
        )

    def score_po(self, p, o, s=None):
        if s is None:
            s = self.get_s_embedder().embed_all()
        else:
            s = self.get_s_embedder().embed(s)
        s_means, s_sigmas = s["means"], s["sigmas"]
        o = self.get_o_embedder().embed(o)
        o_means, o_sigmas = o["means"], o["sigmas"]
        p = self.get_p_embedder().embed(p + self.dataset.num_relations())
        p_means, p_sigmas = p["means"], p["sigmas"]
        # you score 'inverse' as it is reciprocal training
        return self._scorer.score_emb(
            o_means,
            p_means,
            s_means,
            combine="sp*"
        )
        return self._scorer.score_emb(o, p, s, combine="sp*")

    def score_so(self, s, o, p=None):
        raise Exception("This is a reciprocal model and cannot score relations.")

    def score_sp_po(
            self,
            s: torch.Tensor,
            p: torch.Tensor,
            o: torch.Tensor,
            entity_subset: torch.Tensor = None
    ) -> torch.Tensor:
        s = self.get_s_embedder().embed(s)
        s_means, s_sigmas = s["means"], s["sigmas"]
        p_inv = self.get_p_embedder().embed(p + self.dataset.num_relations())
        p_inv_means, p_inv_vars = p_inv["means"], p_inv["sigmas"]
        p = self.get_p_embedder().embed(p)
        p_means, p_sigmas = p["means"], p["sigmas"]
        o = self.get_o_embedder().embed(o)
        o_means, o_sigmas = o["means"], o["sigmas"]

        if self.get_s_embedder() is self.get_o_embedder():
            if entity_subset is not None:
                all_entities = self.get_s_embedder().embed(entity_subset)
            else:
                all_entities = self.get_s_embedder().embed_all()
            all_entities_means, all_entities_sigmas = \
                all_entities["means"], all_entities[
                "sigmas"]
            sp_scores = self._scorer.score_emb(
                s_means,
                p_means,
                all_entities_means,
                combine="sp_")
            po_scores = self._scorer.score_emb(
                o_means,
                p_inv_means,
                all_entities_means,
                combine="sp_"
            )
        else:
            raise Exception(
                "Using different embedders for s and o not supported for this model."
            )
        return torch.cat((sp_scores, po_scores), dim=1)
