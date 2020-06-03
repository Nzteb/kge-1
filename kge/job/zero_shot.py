import torch

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeModel
from kge.job import TrainingJob, EvaluationJob
from kge.misc import kge_base_dir
from kge.util.io import load_checkpoint

from typing import Dict, Union, Optional
from os import path


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
        dataset = dataset.create(self.config, preload_data=False)
        self.config.set("dataset.files.entity_ids.filename", "all_entity_ids.del")
        dataset.config = self.config
        self.dataset = dataset
        self.config.check("zero_shot.obtain_seen_model", ["load", "train"])
        self.obtain_seen_model = self.config.get("zero_shot.obtain_seen_model")

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
        else:
            raise ValueError("zero_shot.type")

    def run(self) -> dict:
        """Run zero-shot protocol."""
        seen_model = self.training_phase()
        full_model = self.auxiliary_phase(seen_model)
        self.evaluation_phase(full_model)

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
        self.config.set("dataset.files.entity_ids.filename", "all_entity_ids")
        eval_job = EvaluationJob.create(
            config=self.config,
            dataset=self.dataset,
            model=full_model
        )
        eval_job.run()

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

        # add the new set of entities to the model
        embedder.add_embeddings(num_unseen)
        # freeze the seen embeddings
        embedder._embeddings.weight.requires_grad = False
        # there is no validation set in this setting
        # accordingly, use the hyperparameters of the seen model as
        # this is the best guess
        seen_config = seen_model.config
        seen_model.dataset = self.dataset
        aux_config = seen_config.clone()
        aux_config.folder = self.config.folder
        aux_config.log_folder = self.config.folder
        aux_config.set("valid.every", 0)
        aux_config.set("train.split", "aux")
        aux_config.set(
            "dataset.files.entity_ids.filename",
            self.config.get("dataset.files.entity_ids.filename")
        )
        aux_config.log(
            "Training on auxiliary set while holding seen embeddings constant."
        )
        job = TrainingJob.create(
            config=aux_config, dataset=self.dataset, model=seen_model, parent_job=self
        )
        job.run()
        return job.model


