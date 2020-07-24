#!/usr/bin/env python
"""Preprocess a KGE dataset into a the format expected by libkge.

Call as `preprocess.py --folder <name>`. The original dataset should be stored in
subfolder `name` and have files "train.txt", "valid.txt", and "test.txt". Each file
contains one SPO triple per line, separated by tabs.

During preprocessing, each distinct entity name and each distinct relation name
is assigned an index (dense). The index-to-object mapping is stored in files
"entity_ids.del" and "relation_ids.del", resp. The triples (as indexes) are stored in
files "train.del", "valid.del", and "test.del". Additionally, the splits
"train_sample.del" (a random subset of train) and "valid_without_unseen.del" and
"test_without_unseen.del" are stored. The "test/valid_without_unseen.del" files are
subsets of "valid.del" and "test.del" resp. where all triples containing entities
or relations not existing in "train.del" have been filtered out.

Metadata information is stored in a file "dataset.yaml".

"""

import argparse
import yaml
import os.path
import numpy as np

from util import analyze_raw_splits
from util import process_split
from util import store_map
from util import process_obj_meta
from util import write_split_meta
from util import RawDataset
from util import RawSplit
from util import write_dataset_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--order_sop", action="store_true")
    args = parser.parse_args()

    print(f"Preprocessing {args.folder}...")

    raw_train = RawSplit(
        key="train",
        file="train.txt",
        create_sample=True,
        meta={"split_type": "train"}
    )

    raw_valid = RawSplit(
        key="valid",
        file="valid.txt",
        create_filtered=True,
        filter_with=raw_train,
        meta={"split_type": "valid"}
    )
    raw_test = RawSplit(
        key="test",
        file="test.txt",
        create_filtered=True,
        filter_with=raw_train,
        meta={"split_type: test"}
    )

    # read data and collect entities and relations; additionally processes metadata
    dataset: RawDataset = analyze_raw_splits(
        raw_splits=[raw_train, raw_valid, raw_test],
        folder=args.folder,
        order_sop=args.order_sop
    )


    # update dataset config with derived splits
    write_split_meta(
        [raw_train, raw_valid, raw_test], dataset.config,
    )

    # write out triples using indexes
    # process and write splits derived from train
    process_split(
        "train",
        dataset,
        file_name=splits["train"]["file_name"],
        file_key=splits["train"]["file_key"],
        order_sop=args.order_sop,
        create_sample=True,
        sample_size=dataset.raw_split_sizes["valid"],
        sample_file=splits_samples["train"]["file_name"],
        sample_key=splits_samples["train"]["file_key"],
    )

    # process and write splits derived from valid/test
    for split in ["valid", "test"]:
        process_split(
            split,
            dataset,
            file_name=splits[split]["file_name"],
            file_key=splits[split]["file_key"],
            order_sop=args.order_sop,
            create_filtered=True,
            filtered_file=splits_wo_unseen[split]["file_name"],
            filtered_key=splits_wo_unseen[split]["file_key"],
            filtered_include_ent=dataset.entities_in_split["train"],
            filtered_include_rel=dataset.relations_in_split["train"],
        )

    # update config with entity string files
    for string in string_files.keys():
        if os.path.exists(os.path.join(args.folder, string_files[string])):
            dataset.config[f"files.{string}.filename"] = string_files.get(string)
            dataset.config[f"files.{string}.type"] = "idmap"

    # finally, write the dataset.yaml file
    write_dataset_config(dataset.config, args.folder)
