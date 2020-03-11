import itertools
import os
import math
import time
from collections import defaultdict

from dataclasses import dataclass

import torch
import torch.utils.data

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeModel

from kge.util import KgeLoss, KgeOptimizer, KgeSampler, KgeLRScheduler
from typing import Any, Callable, Dict, List, Optional
import kge.job.util

SLOTS = [0, 1, 2]
S, P, O = SLOTS


class TrainingJob(Job):
    """Abstract base job to train a single model with a fixed set of hyperparameters.

    Also used by jobs such as :class:`SearchJob`.

    Subclasses for specific training methods need to implement `_prepare` and
    `_process_batch`.

    """

    def __init__(
        self, config: Config, dataset: Dataset, parent_job: Job = None
    ) -> None:
        from kge.job import EvaluationJob

        super().__init__(config, dataset, parent_job)
        self.model: KgeModel = KgeModel.create(config, dataset)
        self.optimizer = KgeOptimizer.create(config, self.model)
        self.lr_scheduler, self.metric_based_scheduler = KgeLRScheduler.create(
            config, self.optimizer
        )
        self.loss = KgeLoss.create(config)
        self.abort_on_nan: bool = config.get("train.abort_on_nan")
        self.batch_size: int = config.get("train.batch_size")
        self.device: str = self.config.get("job.device")
        valid_conf = config.clone()
        valid_conf.set("job.type", "eval")
        valid_conf.set("eval.data", "valid")
        valid_conf.set("eval.trace_level", self.config.get("valid.trace_level"))
        self.valid_job = EvaluationJob.create(
            valid_conf, dataset, parent_job=self, model=self.model
        )
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.trace_batch: bool = self.config.get("train.trace_level") == "batch"
        self.epoch: int = 0
        self.valid_trace: List[Dict[str, Any]] = []
        self.is_prepared = False
        self.model.train()

        # attributes filled in by implementing classes
        self.loader = None
        self.num_examples = None
        self.type_str: Optional[str] = None

        #: Hooks run after training for an epoch.
        #: Signature: job, trace_entry
        self.post_epoch_hooks: List[Callable[[Job, Dict[str, Any]], Any]] = []

        #: Hooks run before starting a batch.
        #: Signature: job
        self.pre_batch_hooks: List[Callable[[Job], Any]] = []

        #: Hooks run before outputting the trace of a batch. Can modify trace entry.
        #: Signature: job, trace_entry
        self.post_batch_trace_hooks: List[Callable[[Job, Dict[str, Any]], Any]] = []

        #: Hooks run before outputting the trace of an epoch. Can modify trace entry.
        #: Signature: job, trace_entry
        self.post_epoch_trace_hooks: List[Callable[[Job, Dict[str, Any]], Any]] = []

        #: Hooks run after a validation job.
        #: Signature: job, trace_entry
        self.post_valid_hooks: List[Callable[[Job, Dict[str, Any]], Any]] = []

        #: Hooks run after training
        #: Signature: job, trace_entry
        self.post_train_hooks: List[Callable[[Job, Dict[str, Any]], Any]] = []

        if self.__class__ == TrainingJob:
            for f in Job.job_created_hooks:
                f(self)

    @staticmethod
    def create(
        config: Config, dataset: Dataset, parent_job: Job = None
    ) -> "TrainingJob":
        """Factory method to create a training job."""
        if config.get("train.type") == "KvsAll":
            return TrainingJobKvsAll(config, dataset, parent_job)
        elif config.get("train.type") == "negative_sampling":
            return TrainingJobNegativeSampling(config, dataset, parent_job)
        elif config.get("train.type") == "1vsAll":
            return TrainingJob1vsAll(config, dataset, parent_job)
        elif config.get("train.type") == "1vsAllProbab":
            return TrainingJob1vsAllProbab(config, dataset, parent_job)
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError("train.type")

    def run(self) -> None:
        """Start/resume the training job and run to completion."""
        self.config.log("Starting training...")
        checkpoint_every = self.config.get("train.checkpoint.every")
        checkpoint_keep = self.config.get("train.checkpoint.keep")
        metric_name = self.config.get("valid.metric")
        patience = self.config.get("valid.early_stopping.patience")
        while True:
            # checking for model improvement according to metric_name
            # and do early stopping and keep the best checkpoint
            if (
                len(self.valid_trace) > 0
                and self.valid_trace[-1]["epoch"] == self.epoch
            ):
                best_index = max(
                    range(len(self.valid_trace)),
                    key=lambda index: self.valid_trace[index][metric_name],
                )
                if best_index == len(self.valid_trace) - 1:
                    self.save(self.config.checkpoint_file("best"))
                if (
                    patience > 0
                    and len(self.valid_trace) > patience
                    and best_index < len(self.valid_trace) - patience
                ):
                    self.config.log(
                        "Stopping early ({} did not improve over best result ".format(
                            metric_name
                        )
                        + "in the last {} validation runs).".format(patience)
                    )
                    break
                if self.epoch > self.config.get(
                    "valid.early_stopping.min_threshold.epochs"
                ) and self.valid_trace[best_index][metric_name] < self.config.get(
                    "valid.early_stopping.min_threshold.metric_value"
                ):
                    self.config.log(
                        "Stopping early ({} did not achieve min treshold after {} epochs".format(
                            metric_name, self.epoch
                        )
                    )
                    break

            # should we stop?
            if self.epoch >= self.config.get("train.max_epochs"):
                self.config.log("Maximum number of epochs reached.")
                break

            # start a new epoch
            self.epoch += 1
            self.config.log("Starting epoch {}...".format(self.epoch))
            trace_entry = self.run_epoch()
            for f in self.post_epoch_hooks:
                f(self, trace_entry)
            self.config.log("Finished epoch {}.".format(self.epoch))

            # update model metadata
            self.model.meta["train_job_trace_entry"] = self.trace_entry
            self.model.meta["train_epoch"] = self.epoch
            self.model.meta["train_config"] = self.config
            self.model.meta["train_trace_entry"] = trace_entry

            # validate
            if (
                self.config.get("valid.every") > 0
                and self.epoch % self.config.get("valid.every") == 0
            ):
                self.valid_job.epoch = self.epoch
                trace_entry = self.valid_job.run()
                self.valid_trace.append(trace_entry)
                for f in self.post_valid_hooks:
                    f(self, trace_entry)
                self.model.meta["valid_trace_entry"] = trace_entry

                # metric-based scheduler step
                if self.metric_based_scheduler:
                    self.lr_scheduler.step(trace_entry[metric_name])

            # epoch-based scheduler step
            if self.lr_scheduler and not self.metric_based_scheduler:
                self.lr_scheduler.step(self.epoch)

            # create checkpoint and delete old one, if necessary
            self.save(self.config.checkpoint_file(self.epoch))
            if self.epoch > 1:
                delete_checkpoint_epoch = -1
                if checkpoint_every == 0:
                    # do not keep any old checkpoints
                    delete_checkpoint_epoch = self.epoch - 1
                elif (self.epoch - 1) % checkpoint_every != 0:
                    # delete checkpoints that are not in the checkpoint.every schedule
                    delete_checkpoint_epoch = self.epoch - 1
                elif checkpoint_keep > 0:
                    # keep a maximum number of checkpoint_keep checkpoints
                    delete_checkpoint_epoch = (
                        self.epoch - 1 - checkpoint_every * checkpoint_keep
                    )
                if delete_checkpoint_epoch > 0:
                    if os.path.exists(
                        self.config.checkpoint_file(delete_checkpoint_epoch)
                    ):
                        self.config.log(
                            "Removing old checkpoint {}...".format(
                                self.config.checkpoint_file(delete_checkpoint_epoch)
                            )
                        )
                        os.remove(self.config.checkpoint_file(delete_checkpoint_epoch))
                    else:
                        self.config.log(
                            "Could not delete old checkpoint {}, does not exits.".format(
                                self.config.checkpoint_file(delete_checkpoint_epoch)
                            )
                        )

        for f in self.post_train_hooks:
            f(self, trace_entry)
        self.trace(event="train_completed")

    def save(self, filename) -> None:
        """Save current state to specified file"""
        self.config.log("Saving checkpoint to {}...".format(filename))
        torch.save(
            {
                "type": "train",
                "config": self.config,
                "epoch": self.epoch,
                "valid_trace": self.valid_trace,
                "model": self.model.save(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "job_id": self.job_id,
            },
            filename,
        )

    def load(self, filename: str) -> str:
        """Load job state from specified file.

        Returns job id of the job that created the checkpoint."""
        self.config.log("Loading checkpoint from {}...".format(filename))
        checkpoint = torch.load(filename, map_location="cpu")
        if "model" in checkpoint:
            # new format
            self.model.load(checkpoint["model"])
        else:
            # old format (deprecated, will eventually be removed)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.valid_trace = checkpoint["valid_trace"]
        self.model.train()
        return checkpoint.get("job_id")

    def resume(self, checkpoint_file: str = None) -> None:
        if checkpoint_file is None:
            last_checkpoint = self.config.last_checkpoint()
            if last_checkpoint is not None:
                checkpoint_file = self.config.checkpoint_file(last_checkpoint)

        if checkpoint_file is not None:
            self.resumed_from_job_id = self.load(checkpoint_file)
            self.trace(
                event="job_resumed", epoch=self.epoch, checkpoint_file=checkpoint_file
            )
            self.config.log(
                "Resumed from {} of job {}".format(
                    checkpoint_file, self.resumed_from_job_id
                )
            )
        else:
            self.config.log("No checkpoint found, starting from scratch...")

    def run_epoch(self) -> Dict[str, Any]:
        "Runs an epoch and returns a trace entry."

        # prepare the job is not done already
        if not self.is_prepared:
            self._prepare()
            self.model.prepare_job(self)  # let the model add some hooks
            self.is_prepared = True

        # variables that record various statitics
        sum_loss = 0.0
        sum_penalty = 0.0
        sum_penalties = defaultdict(lambda: 0.0)
        epoch_time = -time.time()
        prepare_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        optimizer_time = 0.0

        # process each batch
        for batch_index, batch in enumerate(self.loader):
            for f in self.pre_batch_hooks:
                f(self)

            # process batch (preprocessing + forward pass + backward pass on loss)
            self.optimizer.zero_grad()
            batch_result: TrainingJob._ProcessBatchResult = self._process_batch(
                batch_index, batch
            )
            sum_loss += batch_result.avg_loss * batch_result.size

            # determine penalty terms (forward pass)
            batch_forward_time = batch_result.forward_time - time.time()
            penalties_torch = self.model.penalty(
                epoch=self.epoch,
                batch_index=batch_index,
                num_batches=len(self.loader),
                batch=batch,
            )
            batch_forward_time += time.time()

            # backward pass on penalties
            batch_backward_time = batch_result.backward_time - time.time()
            penalty = 0.0
            for index, (penalty_key, penalty_value_torch) in enumerate(penalties_torch):
                penalty_value_torch.backward()
                penalty += penalty_value_torch.item()
                sum_penalties[penalty_key] += penalty_value_torch.item()
            sum_penalty += penalty
            batch_backward_time += time.time()

            # determine full cost
            cost_value = batch_result.avg_loss + penalty

            # abort on nan
            if self.abort_on_nan and math.isnan(cost_value):
                raise FloatingPointError("Cost became nan, aborting training job")

            # TODO # visualize graph
            # if (
            #     self.epoch == 1
            #     and batch_index == 0
            #     and self.config.get("train.visualize_graph")
            # ):
            #     from torchviz import make_dot

            #     f = os.path.join(self.config.folder, "cost_value")
            #     graph = make_dot(cost_value, params=dict(self.model.named_parameters()))
            #     graph.save(f"{f}.gv")
            #     graph.render(f)  # needs graphviz installed
            #     self.config.log("Exported compute graph to " + f + ".{gv,pdf}")

            # print memory stats
            if self.epoch == 1 and batch_index == 0:
                if self.device.startswith("cuda"):
                    self.config.log(
                        "CUDA memory after first batch: allocated={:14,} "
                        "cached={:14,} max_allocated={:14,}".format(
                            torch.cuda.memory_allocated(self.device),
                            torch.cuda.memory_cached(self.device),
                            torch.cuda.max_memory_allocated(self.device),
                        )
                    )

            # update parameters
            batch_optimizer_time = -time.time()
            self.optimizer.step()
            batch_optimizer_time += time.time()

            # tracing/logging
            if self.trace_batch:
                batch_trace = {
                    "type": self.type_str,
                    "scope": "batch",
                    "epoch": self.epoch,
                    "batch": batch_index,
                    "size": batch_result.size,
                    "batches": len(self.loader),
                    "avg_loss": batch_result.avg_loss,
                    "penalties": [p.item() for k, p in penalties_torch],
                    "penalty": penalty,
                    "cost": cost_value,
                    "prepare_time": batch_result.prepare_time,
                    "forward_time": batch_forward_time,
                    "backward_time": batch_backward_time,
                    "optimizer_time": batch_optimizer_time,
                }
                for f in self.post_batch_trace_hooks:
                    f(self, batch_trace)
                self.trace(**batch_trace, event="batch_completed")
            print(
                (
                    "\r"  # go back
                    + "{}  batch{: "
                    + str(1 + int(math.ceil(math.log10(len(self.loader)))))
                    + "d}/{}"
                    + ", avg_loss {:.4E}, penalty {:.4E}, cost {:.4E}, time {:6.2f}s"
                    + "\033[K"  # clear to right
                ).format(
                    self.config.log_prefix,
                    batch_index,
                    len(self.loader) - 1,
                    batch_result.avg_loss,
                    penalty,
                    cost_value,
                    batch_result.prepare_time
                    + batch_forward_time
                    + batch_backward_time
                    + batch_optimizer_time,
                ),
                end="",
                flush=True,
            )

            # update times
            prepare_time += batch_result.prepare_time
            forward_time += batch_forward_time
            backward_time += batch_backward_time
            optimizer_time += batch_optimizer_time

        # all done; now trace and log
        epoch_time += time.time()
        print("\033[2K\r", end="", flush=True)  # clear line and go back

        other_time = (
            epoch_time - prepare_time - forward_time - backward_time - optimizer_time
        )
        trace_entry = dict(
            type=self.type_str,
            scope="epoch",
            epoch=self.epoch,
            batches=len(self.loader),
            size=self.num_examples,
            avg_loss=sum_loss / self.num_examples,
            avg_penalty=sum_penalty / len(self.loader),
            avg_penalties={k: p / len(self.loader) for k, p in sum_penalties.items()},
            avg_cost=sum_loss / self.num_examples + sum_penalty / len(self.loader),
            epoch_time=epoch_time,
            prepare_time=prepare_time,
            forward_time=forward_time,
            backward_time=backward_time,
            optimizer_time=optimizer_time,
            other_time=other_time,
            event="epoch_completed",
        )
        for f in self.post_epoch_trace_hooks:
            f(self, trace_entry)
        trace_entry = self.trace(**trace_entry, echo=True, echo_prefix="  ", log=True)
        return trace_entry

    def _prepare(self):
        """Prepare this job for running.

        Sets (at least) the `loader`, `num_examples`, and `type_str` attributes of this
        job to a data loader, number of examples per epoch, and a name for the trainer,
        repectively.

        Guaranteed to be called exactly once before running the first epoch.

        """
        raise NotImplementedError

    @dataclass
    class _ProcessBatchResult:
        """Result of running forward+backward pass on a batch."""

        avg_loss: float
        size: int
        prepare_time: float
        forward_time: float
        backward_time: float

    def _process_batch(
        self, batch_index: int, batch
    ) -> "TrainingJob._ProcessBatchResult":
        "Run forward and backward pass on batch and return results."
        raise NotImplementedError


class TrainingJobKvsAll(TrainingJob):
    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.label_smoothing = config.check_range(
            "KvsAll.label_smoothing", float("-inf"), 1.0, max_inclusive=False
        )
        if self.label_smoothing < 0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting label_smoothing to 0, "
                    "was set to {}.".format(self.label_smoothing)
                )
                self.label_smoothing = 0
            else:
                raise Exception(
                    "Label_smoothing was set to {}, "
                    "should be at least 0.".format(self.label_smoothing)
                )
        elif self.label_smoothing > 0 and self.label_smoothing <= (
            1.0 / dataset.num_entities()
        ):
            if config.get("train.auto_correct"):
                # just to be sure it's used correctly
                config.log(
                    "Setting label_smoothing to 1/num_entities = {}, "
                    "was set to {}.".format(
                        1.0 / dataset.num_entities(), self.label_smoothing
                    )
                )
                self.label_smoothing = 1.0 / dataset.num_entities()
            else:
                raise Exception(
                    "Label_smoothing was set to {}, "
                    "should be at least {}.".format(
                        self.label_smoothing, 1.0 / dataset.num_entities()
                    )
                )

        config.log("Initializing 1-to-N training job...")
        self.type_str = "KvsAll"

        if self.__class__ == TrainingJobKvsAll:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        # create sp and po label_coords (if not done before)
        train_sp = self.dataset.index("train_sp_to_o")
        train_po = self.dataset.index("train_po_to_s")

        # convert indexes to pytoch tensors: a nx2 keys tensor (rows = keys),
        # an offset vector (row = starting offset in values for corresponding
        # key), a values vector (entries correspond to values of original
        # index)
        #
        # Afterwards, it holds:
        # index[keys[i]] = values[offsets[i]:offsets[i+1]]

        (
            self.train_sp_keys,
            self.train_sp_values,
            self.train_sp_offsets,
        ) = kge.indexing.prepare_index(train_sp)
        (
            self.train_po_keys,
            self.train_po_values,
            self.train_po_offsets,
        ) = kge.indexing.prepare_index(train_po)

        # create dataloader
        self.loader = torch.utils.data.DataLoader(
            range(len(train_sp) + len(train_po)),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            pin_memory=self.config.get("train.pin_memory"),
        )
        self.num_examples = len(train_sp) + len(train_po)

    def _get_collate_fun(self):
        num_sp = len(self.train_sp_keys)

        # create the collate function
        def collate(batch):
            """For a batch of size n, returns a triple of:

            - pairs (nx2 tensor, row = sp or po indexes),
            - label coordinates (position of ones in a batch_size x num_entities tensor)
            - is_sp (vector of size n, 1 if corresponding example_index is sp, 0 if po)
            - triples (all true triples in the batch: needed for weighted penalties only)

            """
            # count how many labels we have
            num_ones = 0
            for example_index in batch:
                if example_index < num_sp:
                    num_ones += self.train_sp_offsets[example_index + 1]
                    num_ones -= self.train_sp_offsets[example_index]
                else:
                    example_index -= num_sp
                    num_ones += self.train_po_offsets[example_index + 1]
                    num_ones -= self.train_po_offsets[example_index]

            # now create the results
            sp_po_batch = torch.zeros([len(batch), 2], dtype=torch.long)
            is_sp = torch.zeros([len(batch)], dtype=torch.long)
            label_coords = torch.zeros([num_ones, 2], dtype=torch.int)
            current_index = 0
            triples = torch.zeros([num_ones, 3], dtype=torch.long)
            for batch_index, example_index in enumerate(batch):
                is_sp[batch_index] = 1 if example_index < num_sp else 0
                if is_sp[batch_index]:
                    keys = self.train_sp_keys
                    offsets = self.train_sp_offsets
                    values = self.train_sp_values
                    sp_po_col_1, sp_po_col_2, o_s_col = S, P, O
                else:
                    example_index -= num_sp
                    keys = self.train_po_keys
                    offsets = self.train_po_offsets
                    values = self.train_po_values
                    o_s_col, sp_po_col_1, sp_po_col_2 = S, P, O

                sp_po_batch[batch_index,] = keys[example_index]
                start = offsets[example_index]
                end = offsets[example_index + 1]
                size = end - start
                label_coords[current_index : (current_index + size), 0] = batch_index
                label_coords[current_index : (current_index + size), 1] = values[
                    start:end
                ]
                triples[current_index : (current_index + size), sp_po_col_1] = keys[
                    example_index
                ][0]
                triples[current_index : (current_index + size), sp_po_col_2] = keys[
                    example_index
                ][1]
                triples[current_index : (current_index + size), o_s_col] = values[
                    start:end
                ]
                current_index += size

            # all done
            return {
                "sp_po_batch": sp_po_batch,
                "label_coords": label_coords,
                "is_sp": is_sp,
                "triples": triples,
            }

        return collate

    def _process_batch(self, batch_index, batch) -> TrainingJob._ProcessBatchResult:
        # prepare
        prepare_time = -time.time()
        sp_po_batch = batch["sp_po_batch"].to(self.device)
        batch_size = len(sp_po_batch)
        label_coords = batch["label_coords"].to(self.device)
        is_sp = batch["is_sp"]
        sp_indexes = is_sp.nonzero().to(self.device).view(-1)
        po_indexes = (is_sp == 0).nonzero().to(self.device).view(-1)
        labels = kge.job.util.coord_to_sparse_tensor(
            batch_size, self.dataset.num_entities(), label_coords, self.device
        ).to_dense()
        if self.label_smoothing > 0.0:
            # as in ConvE: https://github.com/TimDettmers/ConvE
            labels = (1.0 - self.label_smoothing) * labels + 1.0 / labels.size(1)
        prepare_time += time.time()

        # forward/backward pass (sp)
        loss_value = 0.0
        if len(sp_indexes) > 0:
            forward_time = -time.time()
            scores_sp = self.model.score_sp(
                sp_po_batch[sp_indexes, 0], sp_po_batch[sp_indexes, 1]
            )
            loss_value_sp = self.loss(scores_sp, labels[sp_indexes,]) / batch_size
            loss_value = loss_value_sp.item()
            forward_time += time.time()
            backward_time = -time.time()
            loss_value_sp.backward()
            backward_time += time.time()

        # forward/backward pass (po)
        if len(po_indexes) > 0:
            forward_time = -time.time()
            scores_po = self.model.score_po(
                sp_po_batch[po_indexes, 0], sp_po_batch[po_indexes, 1]
            )
            loss_value_po = self.loss(scores_po, labels[po_indexes,]) / batch_size
            loss_value += loss_value_po.item()
            forward_time += time.time()
            backward_time = -time.time()
            loss_value_po.backward()
            backward_time += time.time()

        # all done
        return TrainingJob._ProcessBatchResult(
            loss_value, batch_size, prepare_time, forward_time, backward_time
        )


class TrainingJobNegativeSampling(TrainingJob):
    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self._sampler = KgeSampler.create(config, "negative_sampling", dataset)
        self.is_prepared = False
        self._implementation = self.config.check(
            "negative_sampling.implementation", ["triple", "all", "batch", "auto"],
        )
        if self._implementation == "auto":
            max_nr_of_negs = max(self._sampler.num_samples)
            if self._sampler.shared:
                self._implementation = "batch"
            elif max_nr_of_negs <= 30:
                self._implementation = "triple"
            elif max_nr_of_negs > 30:
                self._implementation = "batch"
        self._max_chunk_size = self.config.get("negative_sampling.chunk_size")

        config.log(
            "Initializing negative sampling training job with "
            "'{}' scoring function ...".format(self._implementation)
        )
        self.type_str = "negative_sampling"

        if self.__class__ == TrainingJobNegativeSampling:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct dataloader"""

        if self.is_prepared:
            return

        self.num_examples = self.dataset.train().size(0)
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            pin_memory=self.config.get("train.pin_memory"),
        )

        self.is_prepared = True

    def _get_collate_fun(self):
        # create the collate function
        def collate(batch):
            """For a batch of size n, returns a tuple of:

            - triples (tensor of shape [n,3], ),
            - negative_samples (list of tensors of shape [n,num_samples]; 3 elements
              in order S,P,O)
            """

            triples = self.dataset.train()[batch, :].long()
            # labels = torch.zeros((len(batch), self._sampler.num_negatives_total + 1))
            # labels[:, 0] = 1
            # labels = labels.view(-1)

            negative_samples = list()
            for slot in [S, P, O]:
                negative_samples.append(self._sampler.sample(triples, slot))
            return {"triples": triples, "negative_samples": negative_samples}

        return collate

    def _process_batch(self, batch_index, batch) -> TrainingJob._ProcessBatchResult:
        # prepare
        prepare_time = -time.time()
        batch_triples = batch["triples"].to(self.device)
        batch_negative_samples = [
            ns.to(self.device) for ns in batch["negative_samples"]
        ]
        batch_size = len(batch_triples)
        prepare_time += time.time()

        loss_value = 0.0
        forward_time = 0.0
        backward_time = 0.0
        labels = None

        # perform processing of batch in smaller chunks to save memory
        max_chunk_size = (
            self._max_chunk_size if self._max_chunk_size > 0 else batch_size
        )
        for chunk_number in range(math.ceil(batch_size / max_chunk_size)):
            # determine data used for this chunk
            chunk_start = max_chunk_size * chunk_number
            chunk_end = min(max_chunk_size * (chunk_number + 1), batch_size)
            negative_samples = [
                ns[chunk_start:chunk_end, :] for ns in batch_negative_samples
            ]
            triples = batch_triples[chunk_start:chunk_end, :]
            chunk_size = chunk_end - chunk_start

            # process the chunk
            for slot in [S, P, O]:
                num_samples = self._sampler.num_samples[slot]
                if num_samples <= 0:
                    continue

                # construct gold labels: first column corresponds to positives,
                # remaining columns to negatives
                if labels is None or labels.shape != torch.Size(
                    [chunk_size, 1 + num_samples]
                ):
                    prepare_time -= time.time()
                    labels = torch.zeros(
                        (chunk_size, 1 + num_samples), device=self.device
                    )
                    labels[:, 0] = 1
                    prepare_time += time.time()

                # compute corresponding scores
                scores = None
                if self._implementation == "triple":
                    # construct triples
                    prepare_time -= time.time()
                    triples_to_score = triples.repeat(1, 1 + num_samples).view(-1, 3)
                    triples_to_score[:, slot] = torch.cat(
                        (
                            triples[:, [slot]],  # positives
                            negative_samples[slot],  # negatives
                        ),
                        1,
                    ).view(-1)
                    prepare_time += time.time()

                    # and score them
                    forward_time -= time.time()
                    scores = self.model.score_spo(
                        triples_to_score[:, 0],
                        triples_to_score[:, 1],
                        triples_to_score[:, 2],
                        direction="s" if slot == S else ("o" if slot == O else "p"),
                    ).view(chunk_size, -1)
                    forward_time += time.time()
                elif self._implementation == "all":
                    # Score against all possible targets. Creates a score matrix of size
                    # [chunk_size, num_entities] or [chunk_size, num_relations]. All
                    # scores relevant for positive and negative triples are contained in
                    # this score matrix.

                    # compute all scores for slot
                    forward_time -= time.time()
                    if slot == S:
                        all_scores = self.model.score_po(triples[:, P], triples[:, O])
                    elif slot == P:
                        all_scores = self.model.score_so(triples[:, S], triples[:, O])
                    elif slot == O:
                        all_scores = self.model.score_sp(triples[:, S], triples[:, P])
                    else:
                        raise NotImplementedError
                    forward_time += time.time()

                    # determine indexes of relevant scores in scoring matrix
                    prepare_time -= time.time()
                    row_indexes = (
                        torch.arange(chunk_size, device=self.device)
                        .unsqueeze(1)
                        .repeat(1, 1 + num_samples)
                        .view(-1)
                    )  # 000 111 222; each 1+num_negative times (here: 3)
                    column_indexes = torch.cat(
                        (
                            triples[:, [slot]],  # positives
                            negative_samples[slot],  # negatives
                        ),
                        1,
                    ).view(-1)
                    prepare_time += time.time()

                    # now pick the scores we need
                    forward_time -= time.time()
                    scores = all_scores[row_indexes, column_indexes].view(
                        chunk_size, -1
                    )
                    forward_time += time.time()
                elif self._implementation == "batch":
                    # Score against all targets contained in the chunk. Creates a score
                    # matrix of size [chunk_size, unique_entities_in_slot] or
                    # [chunk_size, unique_relations_in_slot]. All scores
                    # relevant for positive and negative triples are contained in this
                    # score matrix.
                    forward_time -= time.time()
                    unique_targets, column_indexes = torch.unique(
                        torch.cat((triples[:, [slot]], negative_samples[slot]), 1).view(
                            -1
                        ),
                        return_inverse=True,
                    )

                    # compute scores for all unique targets for slot
                    if slot == S:
                        all_scores = self.model.score_po(
                            triples[:, P], triples[:, O], unique_targets
                        )
                    elif slot == P:
                        all_scores = self.model.score_so(
                            triples[:, S], triples[:, O], unique_targets
                        )
                    elif slot == O:
                        all_scores = self.model.score_sp(
                            triples[:, S], triples[:, P], unique_targets
                        )
                    else:
                        raise NotImplementedError
                    forward_time += time.time()

                    # determine indexes of relevant scores in scoring matrix
                    prepare_time -= time.time()
                    row_indexes = (
                        torch.arange(chunk_size, device=self.device)
                        .unsqueeze(1)
                        .repeat(1, 1 + num_samples)
                        .view(-1)
                    )  # 000 111 222; each 1+num_negative times (here: 3)
                    prepare_time += time.time()

                    # now pick the scores we need
                    forward_time -= time.time()
                    scores = all_scores[row_indexes, column_indexes].view(
                        chunk_size, -1
                    )
                    forward_time += time.time()

                # compute chunk loss (concluding the forward pass of the chunk)
                forward_time -= time.time()
                loss_value_torch = (
                    self.loss(scores, labels, num_negatives=num_samples) / batch_size
                )
                loss_value += loss_value_torch.item()
                forward_time += time.time()

                # backward pass for this chunk
                backward_time -= time.time()
                loss_value_torch.backward()
                backward_time += time.time()

        # all done
        return TrainingJob._ProcessBatchResult(
            loss_value, batch_size, prepare_time, forward_time, backward_time
        )


class TrainingJob1vsAll(TrainingJob):
    """Samples SPO pairs and queries sp* and *po, treating all other entities as negative."""

    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.is_prepared = False
        config.log("Initializing spo training job...")
        self.type_str = "1vsAll"

        if self.__class__ == TrainingJob1vsAll:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct dataloader"""

        if self.is_prepared:
            return

        self.num_examples = self.dataset.train().size(0)
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=lambda batch: {"triples": self.dataset.train()[batch, :].long()},
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            pin_memory=self.config.get("train.pin_memory"),
        )

        self.is_prepared = True

    def _process_batch(self, batch_index, batch) -> TrainingJob._ProcessBatchResult:
        # prepare
        prepare_time = -time.time()
        triples = batch["triples"].to(self.device)
        batch_size = len(triples)
        prepare_time += time.time()

        # forward/backward pass (sp)
        forward_time = -time.time()
        scores_sp = self.model.score_sp(triples[:, 0], triples[:, 1])
        loss_value_sp = self.loss(scores_sp, triples[:, 2]) / batch_size
        loss_value = loss_value_sp.item()
        forward_time += time.time()
        backward_time = -time.time()
        loss_value_sp.backward()
        backward_time += time.time()

        # forward/backward pass (po)
        forward_time -= time.time()
        scores_po = self.model.score_po(triples[:, 1], triples[:, 2])
        loss_value_po = self.loss(scores_po, triples[:, 0]) / batch_size
        loss_value += loss_value_po.item()
        forward_time += time.time()
        backward_time -= time.time()
        loss_value_po.backward()
        backward_time += time.time()

        # all done
        return TrainingJob._ProcessBatchResult(
            loss_value, batch_size, prepare_time, forward_time, backward_time
        )


class TrainingJob1vsAllProbab(TrainingJob):
    def __init__(self, config, dataset, parent_job=None):
        super().__init__(config, dataset, parent_job)
        self.is_prepared = False
        config.log("Initializing spo training job...")
        self.type_str = "1vsAllProbab"
        self.num_eps_samples = self.config.get("1vsAllProbab.reparameterize_samples")
        self.elbo_form = self.config.get("1vsAllProbab.elbo_form")
        self.config.check("1vsAllProbab.elbo_form", ["kl", "ent"])
        self.norm_p = self.config.get("1vsAllProbab.norm_p")

        var = self.config.get("1vsAllProbab.prior_variance")
        if var > 0:
            # one global variance for all entities and relations (and coordinates)
            self.learn_reg = False
            self.prior_variance_ent = var
            self.prior_variance_pred = var
        elif var == -1:
            self.learn_reg = True
            self.prior_variance_ent = None
            self.prior_variance_pred = None
        else:
            raise Exception("Wrong parameter for prior_sigma_sq")

        if self.model.get_s_embedder() != self.model.get_o_embedder():
            raise Exception("Training scheme only supports using same embedders")

        if self.__class__ == TrainingJob1vsAllProbab:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct dataloader"""

        if self.is_prepared:
            return

        self.num_examples = self.dataset.train().size(0)
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=self.get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            pin_memory=self.config.get("train.pin_memory"),
        )

        self.is_prepared = True

    def get_collate_fun(self):
        def collate(batch):
            triples = self.dataset.train()[batch, :].long().to(self.device)

            eps_so = torch.randn(
                self.num_eps_samples,
                self.model.dataset.num_entities(),
                self.model.get_s_embedder().dim,
            ).to(self.config.get("job.device"))
            eps_p = torch.randn(
                self.num_eps_samples,
                self.dataset.num_relations() * 2,  # reciprocal relations
                self.model.get_p_embedder().dim,
            ).to(self.config.get("job.device"))

            s_idx = triples[:, 0]
            p_idx = triples[:, 1]
            o_idx = triples[:, 2]

            all_ent_emb = self.model.get_s_embedder().embed_all()
            all_ent_means, all_ent_sigmas = all_ent_emb["means"], all_ent_emb["sigmas"]

            s_means, s_sigmas = all_ent_means[s_idx], all_ent_sigmas[s_idx]
            o_means, o_sigmas = all_ent_means[o_idx], all_ent_sigmas[o_idx]

            all_p_emb = self.model.get_p_embedder().embed_all()
            all_p_means, all_p_sigmas = all_p_emb["means"], all_p_emb["sigmas"]

            p_means, p_sigmas = all_p_means[p_idx], all_p_sigmas[p_idx]
            p_means_inv = all_p_means[p_idx + self.dataset.num_relations()]
            p_sigmas_inv = all_p_sigmas[p_idx + self.dataset.num_relations()]

            return {
                "triples": triples,
                "eps_so": eps_so,
                "eps_p": eps_p,
                "all_ent_means": all_ent_means,
                "all_ent_sigmas": all_ent_sigmas,
                "s_means": s_means,
                "s_sigmas": s_sigmas,
                "o_means": o_means,
                "o_sigmas": o_sigmas,
                "p_means": p_means,
                "p_sigmas": p_sigmas,
                "p_means_inv": p_means_inv,
                "p_sigmas_inv": p_sigmas_inv,
                "all_p_means": all_p_means,
                "all_p_sigmas": all_p_sigmas,
            }

        return collate

    def _process_batch(self, batch_index, batch) -> TrainingJob._ProcessBatchResult:
        # prepare
        prepare_time = -time.time()
        triples = batch["triples"]
        eps_so = batch["eps_so"]
        eps_p = batch["eps_p"]
        batch_size = len(triples)
        s_idx = triples[:, 0]
        p_idx = triples[:, 1]
        o_idx = triples[:, 2]
        all_ent_means, all_ent_sigmas = batch["all_ent_means"], batch["all_ent_sigmas"]
        s_means, s_sigmas = batch["s_means"], batch["s_sigmas"]
        o_means, o_sigmas = batch["o_means"], batch["o_sigmas"]
        p_means, p_sigmas = batch["p_means"], batch["p_sigmas"]
        p_means_inv, p_sigmas_inv = batch["p_means_inv"], batch["p_sigmas_inv"]

        prepare_time += time.time()

        # EM M-step
        if self.learn_reg:
            self.update_prior_variances(batch)

        # forward/backward pass (sp)
        forward_time = -time.time()

        # each example embeddings are reparameterized and scored x-times
        # where x=num_eps_samples
        # (batch_size * num_eps_samples) X num_entities tensor
        scores_sp = self.model._scorer.score_emb(
            s_means.repeat(self.num_eps_samples, 1)
            + s_sigmas.repeat(self.num_eps_samples, 1)
            * eps_so[:, s_idx, :].view(self.num_eps_samples * batch_size, -1),
            p_means.repeat(self.num_eps_samples, 1)
            + p_sigmas.repeat(self.num_eps_samples, 1)
            * eps_p[:, p_idx, :].view(self.num_eps_samples * batch_size, -1),
            all_ent_means.repeat(self.num_eps_samples, 1)
            + all_ent_sigmas.repeat(self.num_eps_samples, 1)
            * eps_so.view(self.num_eps_samples * len(all_ent_means), -1),
            combine="sp*",
        )
        loss_value_sp = self.loss(scores_sp, o_idx.repeat(self.num_eps_samples))
        loss_value_sp = loss_value_sp / (self.num_eps_samples * batch_size)
        loss_value = loss_value_sp.item()

        forward_time += time.time()
        backward_time = -time.time()
        loss_value_sp.backward(retain_graph=True)
        backward_time += time.time()

        # forward/backward pass (po)
        # this is reciprocal training
        # each example embeddings are reparameterized and scored x-times
        # where x=num_eps_samples
        # (batch_size * num_eps_samples) X num_entities tensor
        scores_po = self.model._scorer.score_emb(
            o_means.repeat(self.num_eps_samples, 1)
            + o_sigmas.repeat(self.num_eps_samples, 1)
            * eps_so[:, o_idx, :].view(self.num_eps_samples * batch_size, -1),
            p_means_inv.repeat(self.num_eps_samples, 1)
            + p_sigmas_inv.repeat(self.num_eps_samples, 1)
            * eps_p[:, p_idx + self.dataset.num_relations(), :].view(
                self.num_eps_samples * batch_size, -1
            ),
            all_ent_means.repeat(self.num_eps_samples, 1)
            + all_ent_sigmas.repeat(self.num_eps_samples, 1)
            * eps_so.view(self.num_eps_samples * len(all_ent_means), -1),
            combine="sp*",
        )
        loss_value_po = self.loss(scores_po, s_idx.repeat(self.num_eps_samples))
        loss_value_po = loss_value_po / (self.num_eps_samples * batch_size)
        loss_value += loss_value_po.item()
        forward_time += time.time()
        backward_time -= time.time()
        loss_value_po.backward(retain_graph=True)
        backward_time += time.time()

        # TODO when models are finalized you might want to move penalty computation
        #  also: penalty and loss are accumulated into loss like this in the trace
        #  as the frameworks penalty computation happens downstream in train but we need it here
        #  because it depends on the training and batch and e.g. epsilons
        if self.elbo_form == "kl":
            loss_value += self._process_penalty_kl(batch)
        elif self.elbo_form == "ent":
            loss_value += self._process_penalty_entropy(batch)

        # all done
        return TrainingJob._ProcessBatchResult(
            loss_value, batch_size, prepare_time, forward_time, backward_time
        )

    def _process_penalty_entropy(self, batch):
        eps_so = batch["eps_so"]
        eps_p = batch["eps_p"]
        all_ent_means, all_ent_sigmas = batch["all_ent_means"], batch["all_ent_sigmas"]
        all_p_means, all_p_sigmas = batch["all_p_means"], batch["all_p_sigmas"]

        if not self.learn_reg:
            prior_variance_ent = self.prior_variance_ent
            prior_variance_pred = self.prior_variance_pred
        else:
            embedder_ent = self.model.get_o_embedder()
            embedder_pred = self.model.get_p_embedder()
            prior_variance_ent = embedder_ent.prior_variance.to(self.device)
            prior_variance_pred = embedder_pred.prior_variance.to(self.device)

        norm_p = self.norm_p  # p-norm
        penalties_reg = torch.zeros(1).to(self.config.get("job.device"))
        penalties_entropy = torch.zeros(1).to(self.config.get("job.device"))

        # TODO vectorize as above
        for i in range(self.num_eps_samples):

            params_ent = (all_ent_means + eps_so[i] * all_ent_sigmas).transpose(0, 1)
            params_pred = (all_p_means + eps_p[i] * all_p_sigmas).transpose(0, 1)
            if norm_p % 2 == 1:
                params_ent = torch.abs(params_ent)
                params_pred = torch.abs(params_pred)

            # regularization term of prior distribution (e.g. gaussian for norm_p=2)
            # prior variance is either a scalar or a vector to scale every entity/pred
            # with their particular variance
            penalties_reg += (1 / norm_p) * (
                (params_ent ** norm_p) / prior_variance_ent
            ).sum()

            penalties_reg += (1 / norm_p) * (
                (params_pred ** norm_p) / prior_variance_pred
            ).sum()

        penalties_reg = penalties_reg / self.num_eps_samples
        # entropy term of variational gaussian and prior gaussian
        penalties_entropy -= torch.log(all_ent_sigmas).sum()
        penalties_entropy -= torch.log(all_p_sigmas).sum()
        # scale penalty with size of dataset (2*num triples) to match expectations
        penalties = (penalties_entropy + penalties_reg) / 1 (
            2 * len(self.dataset.train())
        )
        penalties.backward()

        return penalties.item()

    def _process_penalty_kl(self, batch):
        """Closed form KL of variational and prior gaussian distributions"""
        # TODO move to constructor config check
        if self.norm_p != 2:
            raise Exception(
                "The KL-form of the Elbo is computed from a KL-div of gaussians which"
                " corresponds to norm_p=2"
            )

        if not self.learn_reg:
            prior_variance_ent = torch.as_tensor(self.prior_variance_ent).float()
            prior_variance_pred = torch.as_tensor(self.prior_variance_pred).float()
        else:
            embedder_ent = self.model.get_o_embedder()
            embedder_pred = self.model.get_p_embedder()
            prior_variance_ent = embedder_ent.prior_variance.to(self.device)
            prior_variance_pred = embedder_pred.prior_variance.to(self.device)

        prior_sigma_ent = torch.sqrt(prior_variance_ent).to(self.device)
        prior_sigma_pred = torch.sqrt(prior_variance_pred).to(self.device)
        all_ent_means, all_ent_sigmas = batch["all_ent_means"], batch["all_ent_sigmas"]
        all_p_means, all_p_sigmas = batch["all_p_means"], batch["all_p_sigmas"]
        penalties = (
            torch.log(prior_sigma_ent / all_ent_sigmas.transpose(0, 1))
            + (all_ent_sigmas ** 2 + all_ent_means ** 2).transpose(0, 1)
            / (2 * prior_variance_ent)
        ).sum()

        penalties += (
            torch.log(prior_sigma_pred / all_p_sigmas.transpose(0, 1))
            + (all_p_sigmas ** 2 + all_p_means ** 2).transpose(0, 1)
            / (2 * prior_variance_pred)
        ).sum()
        penalties = penalties / (2 * len(self.dataset.train()))
        penalties.backward()

        return penalties.item()

    @torch.no_grad()
    def update_prior_variances(self, batch):
        """Calculate closed form updates for entity/relation specific regularization"""
        all_ent_means, all_ent_sigmas = batch["all_ent_means"], batch["all_ent_sigmas"]
        all_p_means, all_p_sigmas = batch["all_p_means"], batch["all_p_sigmas"]
        embedder_ent = self.model.get_o_embedder()
        embedder_pred = self.model.get_p_embedder()

        # for each entity and relation, update their regularization term (M-step)
        if self.norm_p == 2:
            embedder_ent.prior_variance = (
                (all_ent_means ** 2).sum(axis=1) + (all_ent_sigmas ** 2).sum(axis=1)
            ) / all_ent_means.size(1)

            embedder_pred.prior_variance = (
                (all_p_means ** 2).sum(axis=1) + (all_p_sigmas ** 2).sum(axis=1)
            ) / all_p_means.size(1)
        elif self.norm_p == 3:
            twopi = torch.as_tensor(2 / math.pi).to(self.device)
            err = torch.abs(all_ent_means) / (
                torch.sqrt(torch.tensor(2.0)) * all_ent_sigmas
            )
            err = torch.erf(err).to(self.device)
            embedder_ent.prior_variance = (
                (
                    torch.sqrt(twopi)
                    * (
                        (all_ent_means ** 2 * all_ent_sigmas + 2 * all_ent_sigmas ** 3)
                        * torch.exp(-1 / 2 * (all_ent_means / all_ent_sigmas) ** 2)
                    )
                    + (
                        3 * torch.abs(all_ent_means) * all_ent_sigmas ** 2
                        + torch.abs(all_ent_means) ** 3
                    )
                    * err
                )
                / all_ent_means.size(1)
            ).sum(axis=1)

            err = torch.abs(all_p_means) / (
                torch.sqrt(torch.tensor(2.0)) * all_p_sigmas
            )
            err = torch.erf(err).to(self.device)
            embedder_pred.prior_variance = (
                (
                    torch.sqrt(twopi)
                    * (
                        (all_p_means ** 2 * all_p_sigmas + 2 * all_p_sigmas ** 3)
                        * torch.exp(-1 / 2 * (all_p_means / all_p_sigmas) ** 2)
                    )
                    + (
                        3 * torch.abs(all_p_means) * all_p_sigmas ** 2
                        + torch.abs(all_p_means) ** 3
                    )
                    * err
                )
                / all_p_means.size(1)
            ).sum(axis=1)
