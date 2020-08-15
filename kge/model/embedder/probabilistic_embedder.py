from torch import Tensor
import torch.nn
import torch.nn.functional

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.misc import round_to_points
from kge.util.io import load_checkpoint
from typing import List


class ProbabilisticEmbedder(KgeEmbedder):
    def __init__(
        self, config: Config, dataset: Dataset, configuration_key: str, vocab_size: int
    ):
        super().__init__(config, dataset, configuration_key)

        # read config
        # TODO
        #self.normalize_p = self.get_option("normalize.p")
        #self.normalize_with_grad = self.get_option("normalize.with_grad")
        self.regularize = self.check_option("regularize", ["", "lp"])
        self.sparse = self.get_option("sparse")
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.vocab_size = vocab_size

        round_embedder_dim_to = self.get_option("round_dim_to")
        if len(round_embedder_dim_to) > 0:
            self.dim = round_to_points(round_embedder_dim_to, self.dim)

        ### Probabilistic Extensions ###

        # does not need to be initialized as it is closed-form updated
        if self.config.get("1vsAllProbab.prior_variance_ent") == -1 \
            and self.config.get("1vsAllProbab.prior_variance_pred") == -1:
            self.prior_variance = torch.ones(self.vocab_size)
            self.init_prior_variances()

        # setup means
        self.means = torch.nn.Parameter(torch.empty(self.vocab_size, self.dim))
        # phi*phi = sigma^2
        # parameterization choosen to keep the variance positive
        # e.g. sqrt(phi*phi) = sigma > 0
        self.phi = torch.nn.Parameter(torch.empty(self.vocab_size, self.dim))

        # initialize params
        init_ = self.get_option("initialize")
        try:
            init_args = self.get_option("initialize_args." + init_)
        except KeyError:
            init_args = self.get_option("initialize_args")

        # Automatically set arg a (lower bound) for uniform_ if not given
        if init_ == "uniform_" and "a" not in init_args:
            init_args["a"] = init_args["b"] * -1
            self.set_option("initialize_args.a", init_args["a"], log=True)

        # initialize params
        initialize_path = self.get_option("initialize_path")
        if initialize_path != "":
            checkpoint = load_checkpoint(initialize_path)
            if self.vocab_size == self.dataset.num_relations():
                self.means.data[:] = checkpoint["model"][0]["_relation_embedder.embeddings.weight"]
            elif self.vocab_size == self.dataset.num_entities():
                self.means.data[:] = checkpoint["model"][0]["_entity_embedder.embeddings.weight"]
        else:
            self.initialize(self.means.data, init_, init_args)
            self.initialize(self.phi.data, init_, init_args)

        # TODO handling negative dropout because using it with ax searches for now
        dropout = self.get_option("dropout")
        if dropout < 0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting {}.dropout to 0, "
                    "was set to {}.".format(configuration_key, dropout)
                )
                dropout = 0
        self.dropout = torch.nn.Dropout(dropout)

    def init_prior_variances(self):
        if self.vocab_size == self.dataset.num_relations():
            index = self.dataset.index("train_relation_frequencies")
        else:
            index = self.dataset.index("train_entity_frequencies")
        for obj in index.keys():
            self.prior_variance[obj] = (1 / index[obj])




    def prepare_job(self, job: Job, **kwargs):
        super().prepare_job(job, **kwargs)
        #TODO:
        # if self.normalize_p > 0:
        #
        #     def normalize_embeddings(job):
        #         if self.normalize_with_grad:
        #             self.embeddings.weight = torch.nn.functional.normalize(
        #                 self.embeddings.weight, p=self.normalize_p, dim=-1
        #             )
        #         else:
        #             with torch.no_grad():
        #                 self.embeddings.weight = torch.nn.Parameter(
        #                     torch.nn.functional.normalize(
        #                         self.embeddings.weight, p=self.normalize_p, dim=-1
        #                     )
        #                 )
        #
        #     job.pre_batch_hooks.append(normalize_embeddings)

    def _embed(self, means, sigmas: Tensor) -> Tensor:
        if self.dropout.p > 0:
            means = self.dropout(means)
            sigmas = self.dropout(sigmas)
        return {"means": means, "sigmas": sigmas}

    def _get_sigma(self):
        return torch.abs(self.phi)

    def embed(self, indexes: Tensor) -> Tensor:
        return self._embed(self.means[indexes.long()], self._get_sigma()[indexes.long()])

    def embed_all(self) -> Tensor:
        return self._embed(self.means, self._get_sigma())

    def penalty(self, **kwargs) -> List[Tensor]:
        raise NotImplementedError
        # # TODO factor out to a utility method
        # result = super().penalty(**kwargs)
        # if self.regularize == "" or self.get_option("regularize_weight") == 0.0:
        #     pass
        # elif self.regularize == "lp":
        #     p = (
        #         self.get_option("regularize_args.p")
        #         if self.has_option("regularize_args.p")
        #         else 2
        #     )
        #     if not self.get_option("regularize_args.weighted"):
        #         # unweighted Lp regularization
        #         parameters = self.embeddings.weight
        #         if p % 2 == 1:
        #             parameters = torch.abs(parameters)
        #         result += [
        #             self.get_option("regularize_weight") / p * (parameters ** p).sum()
        #         ]
        #     else:
        #         # weighted Lp regularization
        #         unique_ids, counts = torch.unique(
        #             kwargs["batch"]["triples"][:, kwargs["slot"]], return_counts=True
        #         )
        #         parameters = self.embeddings(unique_ids)
        #         if p % 2 == 1:
        #             parameters = torch.abs(parameters)
        #         result += [
        #             self.get_option("regularize_weight")
        #             / p
        #             * (parameters ** p * counts.float().view(-1, 1)).sum()
        #             # In contrast to unweighted Lp regulariztion, rescaling by number of
        #             # triples is necessary here so that penalty term is correct in
        #             # expectation
        #             / len(kwargs["batch"]["triples"])
        #         ]
        # else:  # unknown regularziation
        #     raise ValueError(f"Invalid value regularize={self.regularize}")
        #
        # return result
