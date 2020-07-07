import torch

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeModel
from kge.job import TrainingJob, EvaluationJob
from kge.misc import kge_base_dir
from kge.util.io import load_checkpoint
from kge.indexing import where_in

from typing import Dict, Union, Optional
from os import path
import numpy as np
from numpy.linalg import inv

S,P,O = 0,1,2


class ZeroShotProtocolJob(Job):
    def __init__(self, config, dataset, parent_job, model):
        super().__init__(config, dataset, parent_job)

        self.config = config
        # ensure that the dataset of this job corresponds to seen and unseen entities
        # as this job is the meta job over the whole dataset
        # dataset.create overwrites the config with values from dataset.yaml
        # e.g. entity_ids corresponds to only one file, however, the dataset has
        # two different entity_ids files, one corresponding to only the seen
        # entities and one corresponding to all entities
        dataset = Dataset.create(self.config, preload_data=False)
        self.config.set("dataset.files.entity_ids.filename", "all_entity_ids.del")
        dataset.config = self.config
        self.dataset = dataset
        self.config.check("zero_shot.obtain_seen_model", ["load", "train"])
        self.obtain_seen_model = self.config.get("zero_shot.obtain_seen_model")
        self.device = self.config.get("job.device")
        self.config.check("zero_shot.eval_type", ["incremental", "all"])

        self.only_eval = self.config.get("zero_shot.only_eval")
        if self.only_eval and self.config.get("zero_shot.full_model_checkpoint") == "":
            raise Exception(
                "If aux + training phase is skipped, you have to provide a full model"
            )

        # all done, run job_created_hooks if necessary
        if self.__class__ == ZeroShotProtocolJob:
            for f in Job.job_created_hooks:
                f(self)

    @staticmethod
    def create(config, dataset, parent_job=None, model=None):
        """Factory method to create an evaluation job """
        from kge.job import ZeroShotFoldInJob
        # create the job
        if config.get("zero_shot.type") == "fold_in":
            return ZeroShotFoldInJob(config, dataset, parent_job=parent_job, model=model)
        elif config.get("zero_shot.type") == "closed_form":
            return ZeroShotClosedFormJob(config, dataset, parent_job=parent_job, model=model)
        else:
            raise ValueError("zero_shot.type")

    def run(self) -> dict:
        """Run zero-shot protocol."""

        eval_type = self.config.get("zero_shot.eval_type")

        if not self.only_eval:
            if eval_type == "all":
                seen_model = self.training_phase()
                full_model = self.auxiliary_phase(seen_model)
                self.evaluation_phase(full_model)
            elif eval_type == "incremental":
                seen_model = self.training_phase()
                full_model = self.auxiliary_phase(seen_model)
                self.incremental_zero_shot_evaluation_phase(full_model)
        else:
            full_checkpoint_file = self.config.get("zero_shot.full_model_checkpoint")
            full_checkpoint_file = path.join(kge_base_dir(), full_checkpoint_file)


            full_checkpoint = load_checkpoint(
                checkpoint_file=full_checkpoint_file
            )
            full_model = KgeModel.create_from(full_checkpoint, dataset=self.dataset)
            self.incremental_zero_shot_evaluation_phase(full_model)



    def training_phase(self):
        """Train a model on the seen entities or load a pre-trained model."""

        if self.obtain_seen_model == "load":
            seen_checkpoint_file = path.join(
                kge_base_dir(),
                self.config.get("zero_shot.seen_checkpoint")
            )
            checkpoint = load_checkpoint(checkpoint_file=seen_checkpoint_file)
            seen_model = KgeModel.create_from(checkpoint)
            return seen_model

        elif self.obtain_seen_model == "train":
            seen_config = self.config.clone()
            # zero-shot test dataset contains unseen entities; cannot be used
            # during training on the seen entities
            seen_dataset = Dataset.create(seen_config, preload_data=False)
            seen_config.set("eval.filter_with_test", False)
            seen_config.folder = path.join(self.config.folder, "seen_model")
            seen_config.set("job.type", "train")
            seen_config.set("dataset.files.entity_ids.filename", "seen_entity_ids.del")
            seen_config.init_folder()
            seen_dataset.config = seen_config
            job = TrainingJob.create(config=seen_config, dataset=seen_dataset)
            job.run()
        return job.model

    def auxiliary_phase(self, seen_model: KgeModel):
        """Use the auxiliary dataset to gather information on unseen entities.

         :param seen_model: a KgeModel which is trained on the seen entities, e.g., the
                            embedder has a vocabulary size of num. seen entities.

         :returns a KgeModel which must be capable of scoring triples based on the whole
                  vocabulary (seen + unseen entities).
         """

        raise NotImplementedError

    def evaluation_phase(self, full_model: KgeModel):
        """Evaluate the model from training- and auxiliary phase on test data.

        :param full_model: A KgeModel with vocabulary size of all entities (seen+unseen)

        """
        self.config.set("eval.split", "test")
        self.config.check("dataset.files.entity_ids.filename", "all_entity_ids.del")
        self.config.set("entity_ranking.filter_splits", ["aux", "train", "valid"])
        self.config.set("entity_ranking.metrics_per.head_and_tail", True)
        eval_job = EvaluationJob.create(
            config=self.config,
            dataset=self.dataset,
            model=full_model
        )
        eval_job.run()

    def incremental_zero_shot_evaluation_phase(self, full_model):
        """An incremental evaluation protocol.

        For every test fact (s,p,o), when s is unseen rank against
        (?,p,o) where ? is all seen entities + s, and rank against (s,p,?) where
        ? is all seen entities + s.

        """

        unseen_entities = [int(idx) for idx in self.dataset.load_map("unseen_entity_ids").keys()]
        seen_entities = [int(idx) for idx in
                           self.dataset.load_map("seen_entity_ids").keys()]

        all = unseen_entities+seen_entities
        unseen_entities = np.array(unseen_entities)
        seen_entities = np.array(seen_entities)
        all = np.array(all)



        #TODO make unseen slot configurable
        unseen_slot = S

        full_model.eval()

        all_ranks_head = []
        all_ranks_tail = []
        count = 0

        # when scored against all entities (seen + unseen) this implementation
        # produces the same results as libKGE
        # this coded can be used to calculate MRR_filt when scored against subsets
        # e.g. scored against seen_entities + current unseen entity
        # note that this code is super naive/slow it just does everything in a loop
        print("Start incremental evaluation")
        for unseen in unseen_entities:
            count += 1

            # the test facts have a fixed slot where the unseen entity can appear
            test = self.dataset.split("test").to(self.config.get("job.device"))
            test_facts = test[test[:, unseen_slot] == unseen]


            for test_fact in test_facts:
                sp = test_fact[:2]
                po = test_fact[1:]

                # score against seen entities + current unseen
                tails = seen_entities
                heads = seen_entities

                for existing_heads in [
                    self.dataset.index("train_po_to_s")[po[0].item(), po[1].item()],
                    self.dataset.index("valid_po_to_s")[po[0].item(), po[1].item()],
                    self.dataset.index("aux_po_to_s")[po[0].item(), po[1].item()],
                    self.dataset.index("test_po_to_s")[po[0].item(), po[1].item()],
                ]:
                    if len(existing_heads):
                        heads = heads[where_in(heads, np.array(existing_heads), not_in=True)]

                # filter out all existing triples
                for existing_tails in [
                    self.dataset.index("train_sp_to_o")[sp[0].item(), sp[1].item()],
                    self.dataset.index("valid_sp_to_o")[sp[0].item(), sp[1].item()],
                    self.dataset.index("aux_sp_to_o")[sp[0].item(), sp[1].item()],
                    self.dataset.index("test_sp_to_o")[sp[0].item(), sp[1].item()],
                ]:
                    if len(existing_tails):
                        tails = tails[where_in(tails, np.array(existing_tails), not_in=True)]

                true_score_tail = full_model.score_spo(
                    test_fact[0].view(1),
                    test_fact[1].view(1),
                    test_fact[2].view(1),
                    direction="o"
                )
                # score all tails
                tails_triples = torch.zeros(len(tails), 3).to(self.config.get("job.device"))
                tails_triples[:, : 2] = sp
                tails_triples[:, 2] = torch.tensor(tails)

                tails_scores = full_model.score_spo(
                    tails_triples[:, 0],
                    tails_triples[:, 1],
                    tails_triples[:, 2],
                    direction="o")

                num_ties = (tails_scores == true_score_tail).sum()
                filtered_rank_tail = (tails_scores > true_score_tail).sum() + 1 + num_ties // 2
                rr_tail = 1 / filtered_rank_tail.float()
                all_ranks_tail.append(rr_tail)

                true_score_head = full_model.score_spo(
                    test_fact[0].view(1),
                    test_fact[1].view(1),
                    test_fact[2].view(1),
                    direction="s"
                )
                # score all heads
                heads_triples = torch.zeros(len(heads), 3).to(self.config.get("job.device"))
                heads_triples[:, 1:] = po
                heads_triples[:, 0] = torch.tensor(heads)

                heads_scores = full_model.score_spo(
                    heads_triples[:, 0],
                    heads_triples[:, 1],
                    heads_triples[:, 2],
                    direction="s")

                num_ties = (heads_scores == true_score_head).sum()
                filtered_rank_head = (heads_scores > true_score_head).sum() + 1 + num_ties // 2
                rr_head = 1 / filtered_rank_head.float()
                all_ranks_head.append(rr_head)

        mrr_head = torch.FloatTensor(all_ranks_head).mean()
        mrr_tail = torch.FloatTensor(all_ranks_tail).mean()
        self.config.log(f"MRR_head:{mrr_head}")
        self.config.log(f"MRR_tail:{mrr_tail}")
        print(f"MRR_head:{mrr_head}")
        print(f"MRR_tail:{mrr_tail}")


    # TODO load/resume
    # def _load(self, checkpoint: Dict):
    #     if checkpoint["type"] not in ["train", "package"]:
    #         raise ValueError("Can only evaluate train and package checkpoints.")
    #     self.resumed_from_job_id = checkpoint.get("job_id")
    #     self.epoch = checkpoint["epoch"]
    #     self.trace(
    #         event="job_resumed", epoch=self.epoch, checkpoint_file=checkpoint["file"]
    #     )
    #
    # @classmethod
    # def create_from(
    #     cls,
    #     checkpoint: Dict,
    #     new_config: Config = None,
    #     dataset: Dataset = None,
    #     parent_job=None,
    #     eval_split: Optional[str] = None,
    # ) -> Job:
    #     """
    #     Creates a Job based on a checkpoint
    #     Args:
    #         checkpoint: loaded checkpoint
    #         new_config: optional config object - overwrites options of config
    #                           stored in checkpoint
    #         dataset: dataset object
    #         parent_job: parent job (e.g. search job)
    #         eval_split: 'valid' or 'test'.
    #                     Defines the split to evaluate on.
    #                     Overwrites split defined in new_config or config of
    #                     checkpoint.
    #
    #     Returns: Evaluation-Job based on checkpoint
    #
    #     """
    #     if new_config is None:
    #         new_config = Config(load_default=False)
    #     if not new_config.exists("job.type") or new_config.get("job.type") != "eval":
    #         new_config.set("job.type", "eval", create=True)
    #     if eval_split is not None:
    #         new_config.set("eval.split", eval_split, create=True)
    #
    #     return super().create_from(checkpoint, new_config, dataset, parent_job)


class ZeroShotFoldInJob(ZeroShotProtocolJob):
    """Train on the auxiliary dataset while holding seen embeddings constant. """

    def __init__(self, config, dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        if self.__class__ == ZeroShotFoldInJob:
            for f in Job.job_created_hooks:
                f(self)

    def auxiliary_phase(self, seen_model):
        if not (seen_model.get_o_embedder() == seen_model.get_s_embedder()):
            raise Exception("Using distinct subject and object embedder not permitted")
        embedder = seen_model.get_o_embedder()
        num_seen = seen_model.dataset.num_entities()
        num_all = len(self.dataset.map_indexes(indexes=None, key="all_entity_ids"))
        num_unseen = num_all - num_seen

        seen_config = seen_model.config
        seen_model.dataset = self.dataset
        seen_model.to(self.device)
        foldin_config = seen_config.clone()
        foldin_config.folder = self.config.folder
        foldin_config.log_folder = self.config.folder
        foldin_config.set("valid.every", 0)
        foldin_config.set("train.split", "aux")
        foldin_config.set(
            "dataset",
            self.config.get("dataset")
        )
        foldin_config.set("job.device", self.config.get("job.device"))
        # create a new model with the full dataset
        foldin_model = KgeModel.create(foldin_config, self.dataset)

        seen_indexes = torch.tensor(
            [int(i) for i in self.dataset.load_map(key="seen_entity_ids").keys()],
            device=self.device,
        )

        foldin_model.get_o_embedder()._embeddings.weight.data[seen_indexes] = (
           seen_model.get_o_embedder()._embeddings.weight.data
        )

        foldin_model.get_p_embedder()._embeddings.weight.data = (
           seen_model.get_p_embedder()._embeddings.weight.data
        )

        if self.config.get("zero_shot.fold_in.freeze"):
            # freeze embeddings of the seen entities
            foldin_model.get_o_embedder().freeze(seen_indexes)

            # freeze relation embeddings
            foldin_model.get_p_embedder().freeze_all()

            foldin_config.log(
                "Training on auxiliary set while holding seen embeddings constant."
            )

        # create new dataset to remain flexible
        self.config.set("dataset.pickle", False)
        foldin_dataset = Dataset.create(self.config, preload_data=False)
        if self.config.get("zero_shot.fold_in.max_triple") > 0:
            self.subset_data(foldin_dataset)

        foldin_epoch = self.config.get("zero_shot.fold_in.num_epoch")
        if foldin_epoch > 0:
            foldin_config.set("train.max_epochs", foldin_epoch)

        job = TrainingJob.create(
            config=foldin_config,
            dataset=foldin_dataset,
            model=foldin_model,
            parent_job=self
        )
        job.run()
        return job.model

    def subset_data(self, dataset):
        unseen_entities = list(self.dataset.load_map("unseen_entity_ids").keys())
        max_triple = self.config.get("zero_shot.fold_in.max_triple")

        aux = self.dataset.split("aux")
        new_aux = torch.zeros(1, 3).int()
        for unseen in unseen_entities:
            facts = aux[aux[:, 0] == int(unseen)][:max_triple]
            new_aux = torch.cat((facts, new_aux), dim=0)
            facts = aux[aux[:, 2] == int(unseen)][:max_triple]
            new_aux = torch.cat((facts, new_aux), dim=0)
        # TODO you seriously don't want to do it like that
        dataset._triples["aux"] = new_aux[:-1]




class ZeroShotClosedFormJob(ZeroShotProtocolJob):
    """Obtain zero-shot embeddings based on distmult in closed-form."""

    def __init__(self, config, dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        if self.__class__ == ZeroShotFoldInJob:
            for f in Job.job_created_hooks:
                f(self)

    def auxiliary_phase(self, seen_model):
        if not (seen_model.get_o_embedder() == seen_model.get_s_embedder()):
            raise Exception("Using distinct subject and object embedder not permitted")

        unseen_entities = list(self.dataset.load_map("unseen_entity_ids").keys())

        seen_indexes = torch.tensor(
            [int(i) for i in self.dataset.load_map(key="seen_entity_ids").keys()],
            device=self.device,
        )

        # create a new model with the full dataset
        full_model = KgeModel.create(seen_model.config, self.dataset)

        # initialize seen embeddings with the trained model's embeddings
        full_model.get_o_embedder()._embeddings.weight.data[seen_indexes] = (
            seen_model.get_o_embedder()._embeddings.weight.data
        )

        full_model.get_p_embedder()._embeddings.weight.data = (
            seen_model.get_p_embedder()._embeddings.weight.data
        )



        with torch.no_grad():
            for unseen in unseen_entities:
                aux = self.dataset.split("aux")
                # collect all facts for this entity
                us_in_head = aux[aux[:, 0] == int(unseen)]
                us_in_tail = aux[aux[:, 1] == int(unseen)]

                # # decrease fact number
                # us_in_head = us_in_head[:5]
                # us_in_tail = us_in_tail[:5]

                relations_head = us_in_head[:, 1]
                relations_head = seen_model.get_p_embedder().embed(relations_head)

                # this are objects, i.e., where unseen is in head
                entities_head = us_in_head[:,2]
                entities_head = seen_model.get_o_embedder().embed(entities_head)

                prod = (entities_head * relations_head).detach().numpy()

                sum_outer_head = np.dot(prod.transpose(), prod)

                relations_tail = us_in_tail[:, 1]
                relations_tail = seen_model.get_p_embedder().embed(relations_tail)

                # this are subjects, i.e., where unseen is in tail
                entities_tail = us_in_tail[:, 0]
                entities_tail = seen_model.get_s_embedder().embed(entities_tail)

                prod = (entities_tail * relations_tail).detach().numpy()

                sum_outer_tail = np.dot(prod.transpose(), prod)

                A = -0.5*(sum_outer_head + sum_outer_tail + np.eye(
                    sum_outer_head.shape[0], sum_outer_head.shape[1])
                          )


                b_head = (entities_head * relations_head).sum(dim=0)
                b_tail = (entities_tail * relations_tail).sum(dim=0)

                b = b_head + b_tail

                mu = np.dot(-2 * b.detach().numpy(),  inv(A))

                full_model.get_o_embedder(
                )._embeddings.weight.data[int(unseen)] = torch.tensor(mu)
                print(f"Obtained embedding for index {unseen}")

                # # try using the average of all neigbhours
                # all = torch.cat((entities_head, entities_tail))
                # mean = torch.mean(all, dim=0).detach()
                #
                # foldin_model.get_o_embedder()._embeddings.weight.data[
                #     int(unseen)] = mean

            if self.config.get("zero_shot.closed_form.normalize"):
                full_model.get_o_embedder()._embeddings.weight.data =\
                    torch.nn.functional.normalize(
                        full_model.get_o_embedder()._embeddings.weight.data,
                        p=2, dim=-1
                    )


        return full_model

