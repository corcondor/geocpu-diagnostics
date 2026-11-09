#!/usr/bin/env python3
"""ACQUIRE: fetch the six approved Hugging Face datasets into NEW Google Drive folders, verified and with provenance.

Runs on a Google Colab CPU runtime (Drive mounted at /content/drive).  It is deliberately NOT meant to run on the owner's PC:
a real (non-dry) run refuses to start off-Colab unless allow_local=True (that switch exists for the unit tests with fake Hub
functions).  Nothing here needs a secret: every repo is public and ungated, and every Hub call passes token=False so a token
that happens to be present in the environment is never sent, read back or written.

Layout written under ``<root>/external_data_v49/`` (root = /content/drive/MyDrive/GeoCPU_v47_fullmodel_20261106):

    <repo with / -> __>/README.md                    the dataset card, verbatim (licence text lives here)
    <repo with / -> __>/<path in the repo>           TRAIN-eligible files only
    <repo with / -> __>/provenance.json              machine-readable record (see build_provenance)
    <repo with / -> __>/PROVENANCE.md                the same, for people
    do_not_train_dev/<repo__>/<path>                 JaQuAD dev files    (decontamination only, never a converter input)
    do_not_train_validation/<repo__>/<path>          HotpotQA validation (decontamination only, never a converter input)
    do_not_train_*/DO_NOT_TRAIN.txt                  marker

Converters must get their input through train_files(root, repo): it can only return files whose role is "train".

Rules enforced (the owner's decisions, see the task text):
  * licence first: before ANY data file of ANY selected dataset is downloaded, every selected dataset card is fetched and its
    front-matter licence must equal the expected one (otherwise LicenseMismatch);
  * revision pinned to the commit sha that HfApi().dataset_info(repo).sha reports at run time (recorded);
  * every file: size == API size and sha256 == LFS sha256; non-LFS files: size == git blob size and git blob id == API blob id
    (sha256 is computed for the record in both cases); a data file that differs from the planning snapshot embedded in
    DATASETS (taken from the saved API JSON) raises UpstreamDrift;
  * nothing is ever overwritten or deleted on Drive: a final file that exists with the wrong size/sha256 raises
    ExistingFileMismatch, a final file that exists with the right ones is skipped, files are written as *.part with open(...,'xb')
    and renamed only after verification, a stale *.part is renamed to *.part.stale-N (never overwritten, never removed),
    provenance files are write-once (identical record: kept; different record: ProvenanceConflict);
  * the heavy transfer goes through a staging directory on the Colab local disk (default /content/_hf_stage_v49, refused if it
    is inside the Drive root), so a bad download never reaches Drive;
  * dry_run=True prints the plan and the byte total, touches no network and writes nothing.

Colab cell:

    !pip install -q huggingface_hub pyarrow
    from google.colab import drive; drive.mount('/content/drive')
    import sys; sys.path.insert(0, '<folder that holds acquire.py>')
    import acquire
    ROOT = '/content/drive/MyDrive/GeoCPU_v47_fullmodel_20261106'
    acquire.acquire(ROOT, dry_run=True)        # prints what would be fetched, ~1.9 GB
    res = acquire.acquire(ROOT)                # the real run; idempotent, safe to re-run after a disconnect
    drive.flush_and_unmount()                  # Drive uploads in the background: let it finish before the runtime is released

The only hosts contacted are huggingface.co and its CDN/xet hosts (*.huggingface.co, *.hf.co), through huggingface_hub; this module
opens no other connection.  (hf_xet, if installed, does its own transfers to *.hf.co and cannot be sandboxed from Python.)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Optional

ACQUIRE_VERSION = "1"
SCHEMA = "v49_acquire_provenance/1"
OWNER_DECISION = "approved: all six (2026-11-09)"
DEFAULT_ROOT = "/content/drive/MyDrive/GeoCPU_v47_fullmodel_20261106"
EXTERNAL_DIRNAME = "external_data_v49"
DEFAULT_STAGE = "/content/_hf_stage_v49"
ALLOWED_HOSTS = ("huggingface.co", "*.huggingface.co", "*.hf.co")
PLANNING_SOURCE = "Hugging Face API JSON saved 2026-11-09 (hf_meta/*.json) + dataset cards"
CHUNK = 8 << 20

# role -> top-level folder (None: inside the dataset folder).  Only role "train" files may ever be trained on.
ROLE_DIR = {"train": None, "card": None, "dev": "do_not_train_dev", "validation": "do_not_train_validation"}
NO_TRAIN_ROLES = ("dev", "validation")

# Hub entry points are resolved lazily so that importing this module (and dry runs) never imports huggingface_hub and the tests
# can replace both with fakes.
HfApi = None
hf_hub_download = None


class AcquireError(Exception):
    pass


class ExistingFileMismatch(AcquireError):
    """A file already exists at the destination with a different size/sha256 (never overwritten)."""


class VerificationError(AcquireError):
    """A downloaded or copied file does not match the size / sha256 / git blob id the Hub reports."""


class UpstreamDrift(VerificationError):
    """The live Hub listing disagrees with the planning snapshot for a data file."""


class LicenseMismatch(AcquireError):
    pass


class ProvenanceConflict(AcquireError):
    pass


class UnsafeLayout(AcquireError):
    pass


# ------------------------------------------------------------------------------------------------ the table
def F(path, size, role, sha256=None, blob=None, pointer_blob=None):
    """One file to fetch.  LFS files carry sha256 (the LFS oid); plain git files carry blob (git blob sha1)."""
    return {"path": path, "size": size, "role": role, "sha256": sha256, "blob": blob, "pointer_blob": pointer_blob}


_APACHE = ("Apache License 2.0", "https://www.apache.org/licenses/LICENSE-2.0")
_BYSA3 = ("Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)", "https://creativecommons.org/licenses/by-sa/3.0/")
_BYSA4 = ("Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)", "https://creativecommons.org/licenses/by-sa/4.0/")
_PENDING = {"status": "pending_checks", "text": "adoption: decided after the checks (the number of rows is not fixed in advance)"}

# Sizes, sha256 (LFS oid) and git blob ids below were taken from the saved Hugging Face API JSON (planning snapshot).
DATASETS = [
    dict(
        key="coding", repo="llm-jp/Synthetic-JP-EN-Coding-Dataset", planning_sha="f3b782fde6b6ebe7606b42bddea68cb75cfbb547",
        licenses=("apache-2.0",), license_name=_APACHE[0], license_url=_APACHE[1], share_alike=False, row_license="Apache-2.0",
        row_source="llm-jp/Synthetic-JP-EN-Coding-Dataset (LLM-jp; subset of Aratako/Synthetic-JP-EN-Coding-Dataset-801k) - Apache-2.0 - https://huggingface.co/datasets/llm-jp/Synthetic-JP-EN-Coding-Dataset",
        language="ja, en", rows_card="113,796 (train)",
        generator_notes=[
            "Card: a subset of Aratako/Synthetic-JP-EN-Coding-Dataset-801k (code SFT data, Japanese/English mixed).",
            "Parent card (Aratako 801k): seed = Magpie-made Aratako/Synthetic-JP-EN-Coding-Dataset-Magpie-69k, expanded with Evol-Instruct-like methods; "
            "instructions were made by nvidia/Nemotron-4-340B-Instruct, microsoft/Phi-3-medium-4k-instruct, mistralai/Mixtral-8x22B-Instruct-v0.1 and "
            "cyberagent/calm3-22b-chat; the parent card says DeepInfra was used for generation.",
            "Parent card: NO post-hoc filtering was applied, 'records of low quality may be included'.",
        ],
        quality_notes=["Many programming languages and both Japanese and English; about 3.6 KB per row on average (406,143,107 bytes / 113,796 rows)."],
        adoption=dict(_PENDING),
        attribution=dict(title="Synthetic-JP-EN-Coding-Dataset (llm-jp/Synthetic-JP-EN-Coding-Dataset)",
                         authors="LLM-jp (Hirokazu Kiyomaru, Takashi Kodama); parent dataset Aratako/Synthetic-JP-EN-Coding-Dataset-801k",
                         source_url="https://huggingface.co/datasets/llm-jp/Synthetic-JP-EN-Coding-Dataset", extra=[]),
        check_names=["Kiyomaru", "Kodama"],
        files=[
            F("README.md", 910, "card", blob="221b1f76afaf4e460edceb3cc88bd4e0c44ced86"),
            F("data/train-00000-of-00001.parquet", 201478456, "train", sha256="aac23bec3167bdc63550c130cdba7cf80bf4e0ddec5259d62eef609ba5910beb", pointer_blob="222be9ffab6977652ce332929009ca65671e996e"),
        ]),
    dict(
        key="tengentoppa", repo="DeL-TaiseiOzaki/Tengentoppa-sft-qwen2.5-32b-reasoning-100k", planning_sha="fa2497452a9b765e8f1e2b7b08e60b276521c26f",
        licenses=("apache-2.0",), license_name=_APACHE[0], license_url=_APACHE[1], share_alike=False, row_license="Apache-2.0",
        row_source="DeL-TaiseiOzaki/Tengentoppa-sft-qwen2.5-32b-reasoning-100k (Qwen2.5-32B-Instruct generated) - Apache-2.0 - https://huggingface.co/datasets/DeL-TaiseiOzaki/Tengentoppa-sft-qwen2.5-32b-reasoning-100k",
        language="ja", rows_card="125,000 (stated on the card; not re-counted here)",
        generator_notes=[
            "Generator: Qwen2.5-32B-Instruct (card). Temperature 0.8 for instructions and 0.7 otherwise, max_tokens 1024, top_p 0.95, repetition_penalty 1.1; "
            "one of 10 personas chosen at random for the system prompt.",
            "Per sample: persona-based instruction -> chain-of-thought answer procedure (reasoning) -> initial_answer -> self-refined answer (refined_answer).",
            "Card caveat (quoted in Japanese): auto-generated; no human check or edit apart from the instruction text "
            "(\u6307\u793a\u6587\u4ee5\u5916\u4eba\u9593\u306b\u3088\u308b\u78ba\u8a8d\u3084\u7de8\u96c6\u306f\u884c\u308f\u308c\u3066\u3044\u307e\u305b\u3093); quality and accuracy are not guaranteed.",
        ],
        quality_notes=["Keys: instruction, reasoning, initial_answer, refined_answer.  About 8 KB per row on average, so many rows will exceed 2,048 characters."],
        adoption=dict(_PENDING),
        attribution=dict(title="Tengentoppa-sft-qwen2.5-32b-reasoning-100k (DeL-TaiseiOzaki/Tengentoppa-sft-qwen2.5-32b-reasoning-100k)",
                         authors="DeL-TaiseiOzaki (rows generated with Qwen2.5-32B-Instruct)",
                         source_url="https://huggingface.co/datasets/DeL-TaiseiOzaki/Tengentoppa-sft-qwen2.5-32b-reasoning-100k", extra=[]),
        check_names=[],
        files=[
            F("README.md", 2778, "card", blob="55474df179e79c0a2f102d889c3c61cb55d89c9c"),
            F("completed_sets.jsonl", 997252762, "train", sha256="80dd010ad1b4213063acb4f9b2188a02783ad6bb546f504a2c8839f6675a4b3d", pointer_blob="dbeb1e49734a8d4127aadca2eecae8a0bc8c9d17"),
        ]),
    dict(
        key="jaquad", repo="SkelterLabsInc/JaQuAD", planning_sha="05600ff310a0970823e70f82f428893b85c71ffe",
        licenses=("cc-by-sa-3.0",), license_name=_BYSA3[0], license_url=_BYSA3[1], share_alike=True, row_license="CC-BY-SA-3.0",
        row_source="JaQuAD (So et al., Skelter Labs) - CC BY-SA 3.0 - https://huggingface.co/datasets/SkelterLabsInc/JaQuAD",
        language="ja", rows_card="train 31,748 questions (9,713 contexts, 691 articles); dev/validation 3,939 (never trained on); test 4,009 is not public; 39,696 in total",
        generator_notes=[
            "Human annotated (crowdsourced): questions and answer spans written by fluent Japanese speakers (natives and non-natives) who were asked for non-trivial questions about a given context.",
            "Contexts are Japanese Wikipedia paragraphs (88.7% of the articles come from the curated featured/good-article lists).",
            "Splits are made from disjoint sets of Wikipedia articles.  Only the train files are trainable; the dev files are fetched ONLY for decontamination.",
        ],
        quality_notes=["Card: most answers are short; every question is assumed answerable from its context; the card says the dataset is 'incomplete yet'.",
                       "Plain git files (no LFS): integrity is checked with size + git blob id; sha256 is computed for the record."],
        adoption=dict(_PENDING),
        attribution=dict(title="JaQuAD: Japanese Question Answering Dataset for Machine Reading Comprehension (SkelterLabsInc/JaQuAD)",
                         authors="ByungHoon So, Kyuhong Byun, Kyungwon Kang, Seongjin Cho (Skelter Labs)",
                         source_url="https://huggingface.co/datasets/SkelterLabsInc/JaQuAD",
                         extra=["Contexts: Japanese Wikipedia article text (CC BY-SA); each article title is kept in the row so the source article stays traceable.",
                                "ShareAlike: the converted JSONL rows must stay under CC BY-SA 3.0 (or a compatible licence) when they are redistributed."]),
        check_names=["ByungHoon So", "Kyuhong Byun", "Kyungwon Kang", "Seongjin Cho"],
        files=[
            F("README.md", 6742, "card", blob="c81f756f80e3bda212f599390195e5393dcb2fa6"),
            F("data/train/jaquad_train_0000.json", 790020, "train", blob="15b82703a5b063607616d1129b506c103be22ed3"),
            F("data/train/jaquad_train_0001.json", 814618, "train", blob="18333166c669abeb549c2f9e9d8cddb4acdbd21a"),
            F("data/train/jaquad_train_0002.json", 843609, "train", blob="bdb658167c10e55d58f839f0e1f1f7d7cdfdb3a4"),
            F("data/train/jaquad_train_0003.json", 658016, "train", blob="752e29d000b72cb6e76ea8061180a85f114e4271"),
            F("data/train/jaquad_train_0004.json", 790560, "train", blob="836f2dca7bbf150f5a20f80dd4d844fa16e4f829"),
            F("data/train/jaquad_train_0005.json", 775597, "train", blob="62bc65a81a4e8b798303f9b566db27524c86a421"),
            F("data/train/jaquad_train_0006.json", 717697, "train", blob="89e8bd28c6d744e25cd1d6aaa3acd277f92934fd"),
            F("data/train/jaquad_train_0007.json", 800496, "train", blob="5f327ecc3f1213c59296307494dbce871d19d220"),
            F("data/train/jaquad_train_0008.json", 752172, "train", blob="750aedf644c55ddf18e6698c492d8a4fc3df50c4"),
            F("data/train/jaquad_train_0009.json", 796964, "train", blob="06f489ce355886e1114bc5260262930b1d8ad380"),
            F("data/train/jaquad_train_0010.json", 797173, "train", blob="664b6993df2b7fc92f1d2b6e36c854ec46b702a6"),
            F("data/train/jaquad_train_0011.json", 785342, "train", blob="bbc87d5253789d5783595621e54f74b4a37ff7f6"),
            F("data/train/jaquad_train_0012.json", 579524, "train", blob="c7cdf4173bdb7a407353ac3f9dd445cc72ffd681"),
            F("data/train/jaquad_train_0013.json", 742690, "train", blob="0e44977533980105a53c682fd0fecf2d3cb7fab8"),
            F("data/train/jaquad_train_0014.json", 728197, "train", blob="ff8d7510f75b8c0a1b5cd4b99656acf349b7c901"),
            F("data/train/jaquad_train_0015.json", 757375, "train", blob="15ef62ebfa58fe7209ef0db228ff34f2f93d90b9"),
            F("data/train/jaquad_train_0016.json", 762747, "train", blob="c7c4e63c14d7a6927624d0bfbe4acaf1d0878dc2"),
            F("data/train/jaquad_train_0017.json", 771769, "train", blob="627936d2f016c20970a06e621e3e0e38e9ce7d96"),
            F("data/train/jaquad_train_0018.json", 807423, "train", blob="cb8453dc3530fe0330280fbbffbc51ee4569cb6e"),
            F("data/train/jaquad_train_0019.json", 795900, "train", blob="5ed94d0ea96f236999126def51c3530aec86b3cc"),
            F("data/train/jaquad_train_0020.json", 775251, "train", blob="7325f45e40b442ca6fab7f9563b4d6cdb0a4b9dc"),
            F("data/train/jaquad_train_0021.json", 803654, "train", blob="0f486278419e03ac0ffe9d0dbeb68f7b00482d35"),
            F("data/train/jaquad_train_0022.json", 855399, "train", blob="a5a9ff12ffef78a65b29b17294bb5ddf4b9e8ea7"),
            F("data/train/jaquad_train_0023.json", 745120, "train", blob="1cdb3849833167f89ea0209a5453dbe13099dbbd"),
            F("data/train/jaquad_train_0024.json", 770649, "train", blob="e7e1f2bd48910da34fe8b005a83b69d4edb4659a"),
            F("data/train/jaquad_train_0025.json", 778422, "train", blob="6e3651824e35a2136eae17259271e6e80753a587"),
            F("data/train/jaquad_train_0026.json", 763484, "train", blob="c72b52dd31d062520e4c674b8a6a4832f3d64e95"),
            F("data/train/jaquad_train_0027.json", 762216, "train", blob="d17e706bf3f253edf83a4b32b6e60d54ed6c678d"),
            F("data/train/jaquad_train_0028.json", 945484, "train", blob="39dab733e8fc61202820f1fa2ba28c8379038887"),
            F("data/train/jaquad_train_0029.json", 245699, "train", blob="d2bfcbc2b2f56423b036e789e88668f9f1e16315"),
            F("data/dev/jaquad_dev_0000.json", 855457, "dev", blob="bdc9a782b7d7f24e9ef6fb4d10e8959ebc7508df"),
            F("data/dev/jaquad_dev_0001.json", 800689, "dev", blob="93a2708ad28959efe78d7a5c669a717debe2c59b"),
            F("data/dev/jaquad_dev_0002.json", 745518, "dev", blob="4fe6060c912d3fc2444ba4c7eafca42548067420"),
            F("data/dev/jaquad_dev_0003.json", 620520, "dev", blob="272703ab78b4a853fb5a6784ef9a2895f2fac3e7"),
        ]),
    dict(
        key="hotpotqa", repo="hotpotqa/hotpot_qa", planning_sha="1908d6afbbead072334abe2965f91bd2709910ab",
        licenses=("cc-by-sa-4.0",), license_name=_BYSA4[0], license_url=_BYSA4[1], share_alike=True, row_license="CC-BY-SA-4.0",
        row_source="HotpotQA (Yang et al., EMNLP 2018) - CC BY-SA 4.0 - https://huggingface.co/datasets/hotpotqa/hotpot_qa",
        language="en", rows_card="distractor: train 90,447; validation 7,405 (never trained on)",
        generator_notes=[
            "Crowdsourced multi-hop question answering over English Wikipedia paragraphs (card: annotations_creators crowdsourced, language_creators found).",
            "Config 'distractor' only (the 'fullwiki' train files are the same blobs and are not fetched).",
        ],
        quality_notes=["English only; about 6.1 KB per row (552,948,795 bytes / 90,447 rows), so rows do NOT fit 2,048 characters as they are: use only as a skeleton or as short derived rows.",
                       "The validation file is fetched ONLY for decontamination (do_not_train_validation/)."],
        adoption=dict(_PENDING),
        attribution=dict(title="HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering (hotpotqa/hotpot_qa)",
                         authors="Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, Christopher D. Manning (arXiv:1809.09600, EMNLP 2018)",
                         source_url="https://huggingface.co/datasets/hotpotqa/hotpot_qa",
                         extra=["Contexts: English Wikipedia paragraphs (CC BY-SA); paragraph titles are kept in the row so the source article stays traceable.",
                                "ShareAlike: the converted JSONL rows must stay under CC BY-SA 4.0 (or a compatible licence) when they are redistributed."]),
        check_names=["Zhilin", "Saizheng", "Bengio", "Salakhutdinov", "Manning"],
        files=[
            F("README.md", 9522, "card", blob="d077a76e02586322b672e66325ee0e7ec4e3d949"),
            F("distractor/train-00000-of-00002.parquet", 165624177, "train", sha256="76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68", pointer_blob="6a5d2fbef658520bd04c00852c1de33030090934"),
            F("distractor/train-00001-of-00002.parquet", 166162479, "train", sha256="713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f", pointer_blob="d52fc5e088d38d834acd60e2751759c90f291fa8"),
            F("distractor/validation-00000-of-00001.parquet", 27452575, "validation", sha256="c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6", pointer_blob="8df32c2341e895698ebbef0a39a055666c15b201"),
        ]),
    dict(
        key="toolace", repo="Team-ACE/ToolACE", planning_sha="6bda777c88d21e5a204703c1ee45597a8fa4f734",
        licenses=("apache-2.0",), license_name=_APACHE[0], license_url=_APACHE[1], share_alike=False, row_license="Apache-2.0",
        row_source="Team-ACE/ToolACE (Liu et al., arXiv:2409.00920) - Apache-2.0 (per card; generator model unverified) - https://huggingface.co/datasets/Team-ACE/ToolACE",
        language="en, zh", rows_card="not stated on the card (size category 10K<n<100K)",
        generator_notes=[
            "ToolACE pipeline (card + paper arXiv:2409.00920): Tool Self-evolution Synthesis (pool of 26,507 APIs), multi-agent Self-guided Dialog Generation, "
            "Dual-Layer Verification (rule checks + model checks).",
            "The generator model is NOT stated on the card, and the paper text saved in hf_meta names no generator model either: UNVERIFIED.",
        ],
        quality_notes=["English and Chinese function-calling dialogues.  Downloaded and checked only: adoption is gated on the generator being verified."],
        adoption={"status": "hold", "text": "adoption: hold (generator unverified)"},
        attribution=dict(title="ToolACE (Team-ACE/ToolACE): ToolACE: Winning the Points of LLM Function Calling",
                         authors="Weiwen Liu, Xu Huang, Xingshan Zeng, et al. (Team-ACE; arXiv:2409.00920)",
                         source_url="https://huggingface.co/datasets/Team-ACE/ToolACE", extra=[]),
        check_names=["Weiwen Liu", "Xingshan Zeng"],
        files=[
            F("README.md", 1873, "card", blob="e42187577a300679b9e59dfd4d715070f05c7e7c"),
            F("data.json", 37154735, "train", sha256="ba12c083fca7e8da48c67ad5b895e495447da7c66e39a2e19742c082e6cb537e", pointer_blob="466498d8464a9a4aa9ce641edcfe6a564fb3a7b2"),
        ]),
    dict(
        key="wizardlm", repo="llm-jp/wizardlm8x22b-logical-math-coding-sft-ja", planning_sha="568b89c1f7e5b81e30096efe16b2506418fa2c4a",
        licenses=("apache-2.0",), license_name=_APACHE[0], license_url=_APACHE[1], share_alike=False, row_license="Apache-2.0",
        row_source="llm-jp/wizardlm8x22b-logical-math-coding-sft-ja (LLM-jp; subset of kanhatakeyama/wizardlm8x22b-logical-math-coding-sft-ja) - Apache-2.0 - https://huggingface.co/datasets/llm-jp/wizardlm8x22b-logical-math-coding-sft-ja",
        language="ja", rows_card="116,422 (train)",
        generator_notes=[
            "Card: subset of kanhatakeyama/wizardlm8x22b-logical-math-coding-sft-ja and kanhatakeyama/wizardlm8x22b-logical-math-coding-sft_additional-ja.",
            "The generator is a WizardLM 8x22B model by the repo/source-dataset names only; this card does not state it.",
            "The owner's own 2026-11-06 materials plan (Drive materials_plan_20261106.md, file id 1Qq2mHpS9fzlk7dpZv8dwjJFZ3cxSq9IL) lists it under 'avoid: obvious mathematical errors'.",
        ],
        quality_notes=["Logical / math / coding SFT, Japanese.  Downloaded and checked only: use it only if the checks show that the error rate is small."],
        adoption={"status": "hold", "text": "adoption: low priority/hold unless the checks show the error rate is small"},
        attribution=dict(title="wizardlm8x22b-logical-math-coding-sft-ja (llm-jp/wizardlm8x22b-logical-math-coding-sft-ja)",
                         authors="LLM-jp (Hirokazu Kiyomaru, Takashi Kodama); source datasets by kanhatakeyama",
                         source_url="https://huggingface.co/datasets/llm-jp/wizardlm8x22b-logical-math-coding-sft-ja", extra=[]),
        check_names=["Kiyomaru", "Kodama"],
        files=[
            F("README.md", 1084, "card", blob="808bd2f4502178f7bb8c289af2b33ef5eab89121"),
            F("data/train-00000-of-00002.parquet", 149685396, "train", sha256="f0f96b9a6d31150d38f096fe095007591233bcca5ddee872a7c0b85ed84632ee", pointer_blob="22fb6540b22f3c3271f4b624404e1861084e494d"),
            F("data/train-00001-of-00002.parquet", 147648584, "train", sha256="f2fd710a79b6901de3d5feee18b27c05ea4c40dc7744d86d5229efe6c7d8dbbf", pointer_blob="298a93a85672d4a5e643e383526b94d37d9407b2"),
        ]),
]


# ------------------------------------------------------------------------------------------------ small helpers
def folder_of(repo: str) -> str:
    return repo.replace("/", "__")


def fmt_bytes(n: int) -> str:
    return f"{n:,} B"


def _safe_rel(path: str) -> str:
    p = PurePosixPath(path)
    if not path or "\\" in path or p.is_absolute() or any(part in ("", ".", "..") for part in p.parts) or str(p) != path:
        raise UnsafeLayout(f"unsafe relative path in table: {path!r}")
    return path


def dest_rel(ds: dict, f: dict) -> str:
    """Destination path of a file, relative to <root>/external_data_v49/ (posix separators)."""
    rf = _safe_rel(f["path"])
    d = folder_of(ds["repo"])
    role_dir = ROLE_DIR[f["role"]]
    return f"{role_dir}/{d}/{rf}" if role_dir else f"{d}/{rf}"


def _is_within(child, parent) -> bool:
    try:
        c, p = os.path.realpath(child), os.path.realpath(parent)
        return os.path.commonpath([c, p]) == p
    except ValueError:                                  # different drives on Windows
        return False


def _g(obj, name, default=None):
    """Attribute-or-key access: the Hub returns objects, test fakes and the saved JSON use dicts."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def validate_table(datasets) -> None:
    seen_repo, seen_dest = set(), set()
    for ds in datasets:
        repo = ds["repo"]
        if repo in seen_repo:
            raise AcquireError(f"duplicate repo in table: {repo}")
        seen_repo.add(repo)
        cards = [f for f in ds["files"] if f["role"] == "card"]
        if len(cards) != 1:
            raise AcquireError(f"{repo}: exactly one card file required, found {len(cards)}")
        if not ds.get("licenses"):
            raise AcquireError(f"{repo}: expected licence ids missing")
        for f in ds["files"]:
            if f["role"] not in ROLE_DIR:
                raise AcquireError(f"{repo}: bad role {f['role']!r}")
            if not isinstance(f["size"], int) or f["size"] <= 0:
                raise AcquireError(f"{repo}:{f['path']}: bad size")
            if bool(f["sha256"]) == bool(f["blob"]):
                raise AcquireError(f"{repo}:{f['path']}: exactly one of sha256 (LFS) / blob (git) is required")
            if f["sha256"] and not re.fullmatch(r"[0-9a-f]{64}", f["sha256"]):
                raise AcquireError(f"{repo}:{f['path']}: bad sha256")
            if f["blob"] and not re.fullmatch(r"[0-9a-f]{40}", f["blob"]):
                raise AcquireError(f"{repo}:{f['path']}: bad git blob id")
            d = dest_rel(ds, f)
            if d in seen_dest:
                raise AcquireError(f"destination collision: {d}")
            seen_dest.add(d)


def select(datasets, only):
    if only is None:
        return list(datasets)
    if isinstance(only, str):
        only = [x for x in only.split(",") if x.strip()]
    out = []
    for o in only:
        o = o.strip()
        m = [d for d in datasets if o in (d["key"], d["repo"], folder_of(d["repo"]))]
        if len(m) != 1:
            raise ValueError(f"unknown dataset {o!r}; known: " + ", ".join(f"{d['key']} ({d['repo']})" for d in datasets))
        if m[0] not in out:
            out.append(m[0])
    return out


def total_bytes(datasets, roles=None) -> int:
    return sum(f["size"] for ds in datasets for f in ds["files"] if roles is None or f["role"] in roles)


def file_digests(path, git_blob: bool = False) -> dict:
    """size, sha256 and (optionally) the git blob sha1, in one streaming pass."""
    size = os.stat(path).st_size
    h = hashlib.sha256()
    g = hashlib.sha1(b"blob %d\0" % size) if git_blob else None
    with open(path, "rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
            if g is not None:
                g.update(b)
    return {"size": size, "sha256": h.hexdigest(), "git_blob_sha1": g.hexdigest() if g is not None else None}


def _digest_errors(dg: dict, exp: dict) -> list:
    errs = []
    if dg["size"] != exp["size"]:
        errs.append(f"size {dg['size']} != expected {exp['size']}")
    if exp["sha256"]:
        if dg["sha256"] != exp["sha256"]:
            errs.append(f"sha256 {dg['sha256']} != expected {exp['sha256']}")
    elif exp["blob"]:
        if dg["git_blob_sha1"] != exp["blob"]:
            errs.append(f"git blob id {dg['git_blob_sha1']} != expected {exp['blob']}")
    else:
        errs.append("nothing to verify against (the Hub gave neither an LFS sha256 nor a git blob id)")
    return errs


def parse_card_licenses(card_text: str):
    """(licence ids lower-cased, front-matter text) from the YAML header of a dataset card; no PyYAML needed."""
    m = re.match(r"^\ufeff?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(\r?\n|$)", card_text, re.S)
    if not m:
        return [], ""
    fm = m.group(1)
    lines = fm.splitlines()
    ids = []
    for i, line in enumerate(lines):
        mm = re.match(r"^license\s*:\s*(.*?)\s*$", line)
        if not mm:
            continue
        v = mm.group(1)
        if v.startswith("["):
            ids.extend(x.strip().strip("'\"") for x in v.strip("[]").split(",") if x.strip())
        elif v:
            ids.append(v.strip("'\""))
        else:
            for nxt in lines[i + 1:]:
                m2 = re.match(r"^\s*-\s*(.+?)\s*$", nxt)
                if not m2:
                    break
                ids.append(m2.group(1).strip("'\""))
    return [x.lower() for x in ids], fm


def card_license_excerpt(card_text: str, limit: int = 8) -> list:
    out = []
    for line in card_text.splitlines():
        if re.search(r"licen[sc]e|\u30e9\u30a4\u30bb\u30f3\u30b9", line, re.I):
            out.append(line.strip()[:300])
            if len(out) >= limit:
                break
    return out


def attribution_lines(ds: dict) -> list:
    a = ds["attribution"]
    lines = [f"Title: {a['title']}",
             f"Author/source: {a['authors']}; {a['source_url']}",
             f"License: {ds['license_name']} <{ds['license_url']}>",
             "Changes: converted to JSONL rows"]
    lines.extend(a.get("extra", []))
    return lines


# ------------------------------------------------------------------------------------------------ Hub access (lazy)
def _hub():
    global HfApi, hf_hub_download
    if HfApi is None or hf_hub_download is None:
        try:
            import huggingface_hub as hh
        except ImportError as e:               # pragma: no cover - environment dependent
            raise AcquireError("huggingface_hub is not installed; run: pip install -q huggingface_hub pyarrow") from e
        if HfApi is None:
            HfApi = hh.HfApi
        if hf_hub_download is None:
            hf_hub_download = hh.hf_hub_download
    return HfApi, hf_hub_download


def _live_listing(info) -> dict:
    out = {}
    for s in (_g(info, "siblings") or []):
        name = _g(s, "rfilename")
        lfs = _g(s, "lfs")
        out[name] = {"size": _g(s, "size"), "blob": _g(s, "blob_id") or _g(s, "blobId"), "is_lfs": lfs is not None,
                     "lfs_sha256": _g(lfs, "sha256") if lfs is not None else None}
    return out


def _expected_for(ds: dict, f: dict, live: dict):
    """(what the downloaded bytes must match = the LIVE Hub values, drift vs the planning snapshot).  Raises for data files that drifted."""
    repo = ds["repo"]
    L = live.get(f["path"])
    if L is None:
        raise UpstreamDrift(f"{repo}:{f['path']} is not in the live Hub listing at the pinned commit")
    drift = []
    if L["size"] != f["size"]:
        drift.append(f"size live {L['size']} vs planning {f['size']}")
    if f["sha256"]:
        if L["lfs_sha256"] != f["sha256"]:
            drift.append(f"lfs sha256 live {L['lfs_sha256']} vs planning {f['sha256']}")
    elif L["blob"] != f["blob"] or L["is_lfs"]:
        drift.append(f"git blob id live {L['blob']} (lfs={L['is_lfs']}) vs planning {f['blob']}")
    if drift and f["role"] != "card":
        raise UpstreamDrift(f"{repo}:{f['path']} differs from the planning snapshot ({'; '.join(drift)}); stop and re-plan, nothing was downloaded")
    exp = {"size": L["size"], "sha256": L["lfs_sha256"] if L["is_lfs"] else None, "blob": None if L["is_lfs"] else L["blob"]}
    return exp, drift


# ------------------------------------------------------------------------------------------------ never-overwrite file ops
def _quarantine(path: str) -> str:
    n = 1
    while os.path.lexists(f"{path}.stale-{n}"):
        n += 1
    dst = f"{path}.stale-{n}"
    os.rename(path, dst)
    return dst


def _copy_hashing(src, dst):
    h = hashlib.sha256()
    n = 0
    with open(src, "rb") as fi, open(dst, "xb") as fo:         # 'x': fails instead of overwriting
        while True:
            b = fi.read(CHUNK)
            if not b:
                break
            fo.write(b)
            h.update(b)
            n += len(b)
        fo.flush()
        try:
            os.fsync(fo.fileno())
        except OSError:
            pass
    return n, h.hexdigest()


def _copy_resume(src, dst):
    """Copy src to dst (never over a finished file).  An existing dst - the '.part' of a dead run - is CONTINUED when its bytes are exactly the first
    bytes of src (compared byte for byte); anything else is moved aside by _quarantine (never deleted) and the copy starts again.  A runtime that
    died after 900 MB of a 1 GB copy therefore costs 100 MB, not 1 GB, and leaves no new '.stale-N' file behind.
    Returns (bytes in dst, sha256 of all of them, bytes that were already there)."""
    h = hashlib.sha256()
    n = resumed = 0
    mode = "xb"
    if os.path.lexists(dst):
        have = os.path.getsize(dst) if os.path.isfile(dst) else -1
        ok = 0 <= have <= os.path.getsize(src)
        if ok:
            with open(src, "rb") as fs, open(dst, "rb") as fd:
                left = have
                while left:
                    a, b = fs.read(min(CHUNK, left)), fd.read(min(CHUNK, left))
                    if not a or a != b:
                        ok = False
                        break
                    h.update(a)
                    n += len(a)
                    left -= len(a)
        if ok:
            mode, resumed = "ab", n
        else:
            _quarantine(str(dst))
            h, n = hashlib.sha256(), 0
    with open(src, "rb") as fi, open(dst, mode) as fo:
        fi.seek(n)
        while True:
            b = fi.read(CHUNK)
            if not b:
                break
            fo.write(b)
            h.update(b)
            n += len(b)
        fo.flush()
        try:
            os.fsync(fo.fileno())
        except OSError:
            pass
    return n, h.hexdigest(), resumed


def _write_new_text(path: Path, text: str) -> None:
    """Create a text file that did not exist; never overwrites."""
    if os.path.lexists(path):
        raise ExistingFileMismatch(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = str(path) + ".part"
    if os.path.lexists(part):
        _quarantine(part)
    with open(part, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.rename(part, path)


class _Ctx:
    pass


def _verify_existing(final: Path, exp: dict, rel: str) -> dict:
    if not os.path.isfile(final):
        raise ExistingFileMismatch(f"{rel} exists but is not a regular file: {final}")
    dg = file_digests(final, git_blob=exp["blob"] is not None)
    errs = _digest_errors(dg, exp)
    if errs:
        raise ExistingFileMismatch(f"{rel} already exists with different content ({'; '.join(errs)}); NOT overwritten. "
                                   f"Use a fresh root or move the old file by hand.")
    return dg


def _stage(ctx, ds: dict, f: dict, sha: str, exp: dict):
    """Make a verified copy of one repo file in the staging dir (local disk).  Returns (path, digests)."""
    local_dir = ctx.stage / folder_of(ds["repo"])
    target = local_dir / f["path"]
    want_git = exp["blob"] is not None
    if os.path.lexists(target):
        dg = file_digests(target, want_git)
        if not _digest_errors(dg, exp):
            ctx.log(f"  staged copy already verified: {f['path']}")
            return target, dg
        if not _is_within(target, ctx.stage):
            raise UnsafeLayout(f"refusing to remove {target}: outside the staging dir")
        ctx.log(f"  stale staged copy of {f['path']} (staging dir only): replacing")
        os.remove(target)
    local_dir.mkdir(parents=True, exist_ok=True)
    ctx.log(f"  downloading {ds['repo']}:{f['path']}  {fmt_bytes(exp['size'])}  @ {sha[:12]}")
    got = ctx.download(repo_id=ds["repo"], filename=f["path"], repo_type="dataset", revision=sha, local_dir=str(local_dir), token=False)
    got = Path(got)
    if not os.path.isfile(got):
        raise VerificationError(f"{ds['repo']}:{f['path']}: hf_hub_download returned {got}, which is not a file")
    dg = file_digests(got, want_git)
    errs = _digest_errors(dg, exp)
    if errs:
        raise VerificationError(f"{ds['repo']}:{f['path']} failed verification ({'; '.join(errs)}); nothing was written to Drive; "
                                f"the bad download stays at {got} and is replaced on the next run")
    return got, dg


def _commit(ctx, staged: Path, dg: dict, final: Path, rel: str) -> None:
    """staged -> <final>.part (never over an existing file) -> verified -> renamed to final."""
    final.parent.mkdir(parents=True, exist_ok=True)
    part = str(final) + ".part"
    reuse = False
    if os.path.lexists(part):
        if os.path.isfile(part) and os.path.getsize(part) == dg["size"] and file_digests(part)["sha256"] == dg["sha256"]:
            reuse = True
            ctx.log(f"  complete {os.path.basename(part)} from an earlier run is correct: finishing it")
    if not reuse:
        had = os.path.lexists(part)
        n, sha, resumed = _copy_resume(staged, part)               # continues a partial of an earlier run, or moves a bad one aside
        if had and resumed:
            ctx.log(f"  continued the partial copy {os.path.basename(part)} of an earlier run after {resumed:,} bytes")
        elif had:
            ctx.log(f"  the partial copy {os.path.basename(part)} of an earlier run did not match the file: moved aside (not deleted), copying again")
        if n != dg["size"] or sha != dg["sha256"]:
            raise VerificationError(f"{rel}: the copy on Drive differs from the verified staged file (kept as {part} for inspection)")
        if ctx.verify_after_copy:
            d2 = file_digests(part)
            if d2["size"] != dg["size"] or d2["sha256"] != dg["sha256"]:
                raise VerificationError(f"{rel}: read-back of the Drive copy does not match (kept as {part} for inspection)")
    if os.path.lexists(final):
        raise ExistingFileMismatch(f"{rel} appeared while it was being written; NOT overwritten (the finished copy stays at {part})")
    os.rename(part, final)


def _ensure_file(ctx, st: dict, f: dict) -> dict:
    ds = st["ds"]
    exp = st["exp"][f["path"]]
    rel = dest_rel(ds, f)
    final = ctx.ext / rel
    t0 = time.time()
    if os.path.lexists(final):
        dg = _verify_existing(final, exp, rel)
        status = "skipped_existing"
        ctx.log(f"  present and verified, skipped: {rel}")
    else:
        staged, dg = _stage(ctx, ds, f, st["sha"], exp)
        _commit(ctx, staged, dg, final, rel)
        status = "downloaded"
        if not ctx.keep_stage and _is_within(staged, ctx.stage):
            os.remove(staged)
        ctx.log(f"  stored: {rel}")
    checks = ["size == Hub size", "sha256 == LFS oid" if exp["sha256"] else "git blob id == Hub blob id (sha256 computed for the record)"]
    return {"path": f["path"], "role": f["role"], "may_train": f["role"] == "train", "dest_rel": rel, "size": dg["size"], "sha256": dg["sha256"],
            "git_blob_sha1": dg["git_blob_sha1"], "storage": "lfs" if exp["sha256"] else "git", "verified_against_hub": checks, "status": status,
            "seconds": round(time.time() - t0, 2),
            "resolve_url": f"https://huggingface.co/datasets/{ds['repo']}/resolve/{st['sha']}/{f['path']}",
            "drift_from_planning_snapshot": st["drift"][f["path"]]}


# ------------------------------------------------------------------------------------------------ provenance
def build_provenance(ctx, st: dict, records: list) -> dict:
    ds = st["ds"]
    by_role = {}
    for r in records:
        by_role[r["role"]] = by_role.get(r["role"], 0) + r["size"]
    names_found = {n: (n in st["card_text"]) for n in ds.get("check_names", [])}
    try:
        from importlib.metadata import version as _ver
        hub_ver = _ver("huggingface_hub")
    except Exception:
        hub_ver = None
    try:
        me = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except Exception:
        me = None
    return {
        "schema": SCHEMA,
        "repo_id": ds["repo"],
        "dataset_key": ds["key"],
        "url": f"https://huggingface.co/datasets/{ds['repo']}",
        "commit_sha": st["sha"],
        "planning_snapshot_commit_sha": ds["planning_sha"],
        "planning_snapshot_source": PLANNING_SOURCE,
        "commit_changed_since_planning": st["sha"] != ds["planning_sha"],
        "retrieved_utc": ctx.now,
        "owner_decision": OWNER_DECISION,
        "license": {"expected_ids": list(ds["licenses"]), "ids_from_card": st["card_ids"], "name": ds["license_name"], "url": ds["license_url"],
                    "card_front_matter": st["card_fm"], "card_licence_lines": card_license_excerpt(st["card_text"]),
                    "card_file": f"{folder_of(ds['repo'])}/README.md", "share_alike": ds["share_alike"]},
        "attribution": {"lines": attribution_lines(ds), "names_found_in_card": names_found},
        "row_defaults": {"source": ds["row_source"], "license": ds["row_license"]},
        "language": ds["language"],
        "rows_per_card": ds["rows_card"],
        "generator_notes": list(ds["generator_notes"]),
        "quality_notes": list(ds["quality_notes"]),
        "adoption": dict(ds["adoption"]),
        "files": records,
        "totals": {"files": len(records), "bytes": sum(r["size"] for r in records), "bytes_by_role": by_role,
                   "train_files": sum(1 for r in records if r["may_train"])},
        "rules_for_converters": ["Read only files with may_train == true (use acquire.train_files).",
                                 "Files with role dev/validation live under do_not_train_*/ and are for decontamination only.",
                                 "Set each row's source and license from row_defaults; keep the attribution lines with the dataset."],
        "tool": {"acquire_version": ACQUIRE_VERSION, "acquire_py_sha256": me, "python": sys.version.split()[0], "huggingface_hub": hub_ver},
        "network": {"allowed_hosts": list(ALLOWED_HOSTS), "token_used": False, "revision_pinned": True},
    }


def _core(prov: dict):
    return {"repo_id": prov["repo_id"], "files": sorted((f["dest_rel"], f["size"], f["sha256"], f["role"]) for f in prov["files"])}


def render_markdown(prov: dict) -> str:
    L = prov["license"]
    out = [f"# Provenance: {prov['repo_id']}", "",
           f"- Repo: {prov['url']}",
           f"- Pinned commit: `{prov['commit_sha']}` (planning snapshot: `{prov['planning_snapshot_commit_sha']}`"
           + (", CHANGED since planning" if prov["commit_changed_since_planning"] else ", unchanged") + ")",
           f"- Retrieved (UTC): {prov['retrieved_utc']}",
           f"- Owner decision: {prov['owner_decision']}",
           f"- Licence: {L['name']} <{L['url']}>; ids on the card: {', '.join(L['ids_from_card'])}",
           f"- Language: {prov['language']}; rows per card: {prov['rows_per_card']}",
           f"- {prov['adoption']['text']}",
           "", "## Licence text from the card (lines mentioning the licence; the full card is README.md in this folder)", "", "```"]
    out.extend(L["card_licence_lines"] or ["(no line mentions the licence besides the front matter)"])
    out += ["", "--- front matter ---", L["card_front_matter"], "```", "", "## Generator and origin notes", ""]
    out.extend(f"- {x}" for x in prov["generator_notes"])
    out += ["", "## Quality notes", ""]
    out.extend(f"- {x}" for x in prov["quality_notes"])
    out += ["", "## Attribution (required; carry it with every converted row/dataset)", ""]
    out.extend(f"- {x}" for x in prov["attribution"]["lines"])
    out += ["", f"Row defaults: source = `{prov['row_defaults']['source']}`; license = `{prov['row_defaults']['license']}`", "",
            "## Files (paths relative to external_data_v49/)", "", "| role | may_train | path | bytes | sha256 |", "|---|---|---|---|---|"]
    for f in prov["files"]:
        out.append(f"| {f['role']} | {str(f['may_train']).lower()} | {f['dest_rel']} | {f['size']} | {f['sha256']} |")
    out += ["", "## Rules for converters", ""]
    out.extend(f"- {x}" for x in prov["rules_for_converters"])
    return "\n".join(out) + "\n"


def _write_provenance(ctx, st: dict, prov: dict) -> str:
    d = ctx.ext / folder_of(st["ds"]["repo"])
    jp, mp = d / "provenance.json", d / "PROVENANCE.md"
    if os.path.lexists(jp):
        old = json.loads(jp.read_text(encoding="utf-8"))
        if _core(old) != _core(prov):
            raise ProvenanceConflict(f"{jp} records different files than this run verified; not overwritten. Compare both and use a fresh root if the data changed.")
        state = "kept"
        if not os.path.lexists(mp):
            _write_new_text(mp, render_markdown(old))
            state = "kept (PROVENANCE.md re-created)"
        return state
    _write_new_text(jp, json.dumps(prov, ensure_ascii=False, indent=1) + "\n")
    if not os.path.lexists(mp):
        _write_new_text(mp, render_markdown(prov))
    return "written"


# ------------------------------------------------------------------------------------------------ dry run / main entry
def _plan(datasets, root=None, ext_dirname: str = EXTERNAL_DIRNAME) -> dict:
    ds_out = []
    for ds in datasets:
        files = []
        for f in ds["files"]:
            rel = dest_rel(ds, f)
            present = None
            if root is not None:
                p = Path(root) / ext_dirname / rel
                present = bool(os.path.isfile(p) and os.path.getsize(p) == f["size"])
            files.append({"role": f["role"], "path": f["path"], "size": f["size"], "dest_rel": rel, "already_present_size_ok": present})
        ds_out.append({"repo": ds["repo"], "key": ds["key"], "planning_sha": ds["planning_sha"], "licenses": list(ds["licenses"]),
                       "bytes": sum(x["size"] for x in files), "files": files})
    return {"datasets": ds_out, "n_files": sum(len(d["files"]) for d in ds_out), "total_bytes": sum(d["bytes"] for d in ds_out),
            "card_bytes": total_bytes(datasets, ("card",)), "train_bytes": total_bytes(datasets, ("train",)),
            "do_not_train_bytes": total_bytes(datasets, NO_TRAIN_ROLES)}


def _dry_run(datasets, root, log, ext_dirname: str = EXTERNAL_DIRNAME) -> dict:
    plan = _plan(datasets, root, ext_dirname)
    log("DRY RUN: no network, nothing is written.  Files that WOULD be fetched (revision = commit sha read from the Hub at run time):")
    for i, d in enumerate(plan["datasets"], 1):
        log(f"{i}. {d['repo']}  [{', '.join(d['licenses'])}]  planning commit {d['planning_sha'][:12]}")
        for f in d["files"]:
            mark = "" if f["already_present_size_ok"] is None else ("  (present, size ok: would be re-verified and skipped)" if f["already_present_size_ok"] else "")
            log(f"     {f['role']:<10} {f['size']:>13,} B  {f['path']}  ->  {ext_dirname}/{f['dest_rel']}{mark}")
        log(f"     subtotal {d['bytes']:,} B")
    tb = plan["total_bytes"]
    log(f"TOTAL: {plan['n_files']} files, {tb:,} bytes = {tb / 1e9:.2f} GB = {tb / 2**30:.2f} GiB "
        f"(train {plan['train_bytes']:,}; do-not-train dev/validation {plan['do_not_train_bytes']:,}; cards {plan['card_bytes']:,})")
    plan["dry_run"] = True
    return plan


def _in_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ or os.path.isdir("/content/sample_data")


def acquire(root, only=None, *, dry_run=False, now_utc=None, datasets=None, stage_dir=None, keep_stage=False,
            verify_after_copy=True, allow_local=False, log=print, ext_dirname: str = EXTERNAL_DIRNAME, pin_existing: bool = True) -> dict:
    """Fetch (or, with dry_run, only plan) the selected datasets into <root>/external_data_v49/.  Idempotent and resumable.

    root        the existing Drive folder (never created here); only NEW sub-folders of it are written.
    only        None = all six, or keys / repo ids / folder names (list or comma string).
    now_utc     retrieval time to record, 'YYYY-MM-DDTHH:MM:SSZ'; read from the clock inside this call when omitted (never at import).
    datasets    override the table (tests only).
    stage_dir   local-disk staging dir (default /content/_hf_stage_v49); must be outside root.
    allow_local real runs refuse to start off-Colab unless this is True (tests only: the owner's PC must not be loaded).
    ext_dirname the folder under root that receives the files (default external_data_v49; the notebook's RAW_SUFFIX makes it a fresh one).
    pin_existing a dataset that already has a provenance.json is fetched at THAT commit, not at the current head of the repo: an upstream edit
                of a README between two runs must not turn a re-run into a refusal.
    """
    table = list(DATASETS if datasets is None else datasets)
    validate_table(table)
    sel = select(table, only)
    if not sel:
        raise ValueError("no dataset selected")
    if dry_run:
        return _dry_run(sel, root, log, ext_dirname)

    if not allow_local and not _in_colab():
        raise AcquireError("refusing to download off Colab: this pipeline must not load the owner's PC (pass allow_local=True only for tests)")
    root = Path(root)
    if str(root).replace("\\", "/").startswith("/content/drive") and not os.path.isdir("/content/drive/MyDrive"):
        raise AcquireError("Google Drive is not mounted: run drive.mount('/content/drive') first")
    if not root.is_dir():
        raise AcquireError(f"root folder does not exist (it is never created here): {root}")
    if now_utc is None:
        now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", now_utc):
        raise ValueError(f"now_utc must look like 2026-11-09T12:00:00Z, got {now_utc!r}")
    if stage_dir is None:
        stage_dir = DEFAULT_STAGE if _in_colab() else tempfile.mkdtemp(prefix="hf_stage_v49_")
    stage = Path(stage_dir)
    if _is_within(stage, root) or _is_within(root, stage):
        raise UnsafeLayout(f"the staging dir {stage} must be outside the Drive root {root}")
    stage.mkdir(parents=True, exist_ok=True)
    biggest = max(f["size"] for d in sel for f in d["files"])
    need = (sum(f["size"] for d in sel for f in d["files"]) if keep_stage else biggest) + (256 << 20)
    free = shutil.disk_usage(stage).free
    if free < need:
        raise AcquireError(f"staging disk has {free:,} B free, need about {need:,} B")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    api_cls, dl = _hub()
    ctx = _Ctx()
    ctx.ext = root / ext_dirname
    ctx.stage, ctx.download, ctx.log, ctx.now = stage, dl, log, now_utc
    ctx.keep_stage, ctx.verify_after_copy = keep_stage, verify_after_copy
    ctx.ext.mkdir(exist_ok=True)
    api = api_cls()

    # ---- phase 1: listing, drift and LICENCE for every selected dataset before any data file is downloaded
    states = []
    for ds in sel:
        pinned = _pinned_commit(ctx.ext, ds) if pin_existing else None
        if pinned:
            log(f"  {ds['repo']}: an earlier run recorded commit {pinned[:12]}; fetching that commit, not the current head")
            info = api.dataset_info(ds["repo"], files_metadata=True, token=False, revision=pinned)
        else:
            info = api.dataset_info(ds["repo"], files_metadata=True, token=False)
        sha = _g(info, "sha")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise VerificationError(f"{ds['repo']}: the Hub did not return a commit sha")
        if pinned and sha != pinned:
            raise VerificationError(f"{ds['repo']}: asked for the recorded commit {pinned} and the Hub answered {sha}")
        live = _live_listing(info)
        st = {"ds": ds, "sha": sha, "exp": {}, "drift": {}}
        for f in ds["files"]:
            st["exp"][f["path"]], st["drift"][f["path"]] = _expected_for(ds, f, live)
        card = next(f for f in ds["files"] if f["role"] == "card")
        card_final = ctx.ext / dest_rel(ds, card)
        if os.path.lexists(card_final):
            _verify_existing(card_final, st["exp"][card["path"]], dest_rel(ds, card))
            raw = card_final.read_bytes()
        else:
            staged, _ = _stage(ctx, ds, card, sha, st["exp"][card["path"]])
            raw = Path(staged).read_bytes()
        st["card_text"] = raw.decode("utf-8", errors="replace")
        st["card_ids"], st["card_fm"] = parse_card_licenses(st["card_text"])
        if set(st["card_ids"]) != set(ds["licenses"]):
            raise LicenseMismatch(f"{ds['repo']}: the card says licence {st['card_ids'] or 'NONE'}, expected {list(ds['licenses'])}; nothing was downloaded")
        log(f"licence ok: {ds['repo']} -> {', '.join(st['card_ids'])}  (commit {sha[:12]})")
        states.append(st)

    # ---- phase 2: the files
    result = {"dry_run": False, "root": str(root), "external_dir": str(ctx.ext), "retrieved_utc": now_utc, "datasets": {}}
    for i, st in enumerate(states, 1):
        ds = st["ds"]
        log(f"[{i}/{len(states)}] {ds['repo']}")
        records = [_ensure_file(ctx, st, f) for f in ds["files"]]
        prov = build_provenance(ctx, st, records)
        state = _write_provenance(ctx, st, prov)
        for role in NO_TRAIN_ROLES:
            if any(r["role"] == role for r in records):
                marker = ctx.ext / ROLE_DIR[role] / "DO_NOT_TRAIN.txt"
                if not os.path.lexists(marker):
                    _write_new_text(marker, f"Files under {ROLE_DIR[role]}/ are {role} data kept ONLY for decontamination. "
                                            "Never train on them and never pass them to a converter.\n")
        result["datasets"][ds["repo"]] = {"commit_sha": st["sha"], "provenance": state, "adoption": ds["adoption"]["text"],
                                          "files": [{"dest_rel": r["dest_rel"], "role": r["role"], "status": r["status"], "size": r["size"], "sha256": r["sha256"]} for r in records]}
    result["total_bytes"] = sum(f["size"] for d in result["datasets"].values() for f in d["files"])
    result["downloaded_bytes"] = sum(f["size"] for d in result["datasets"].values() for f in d["files"] if f["status"] == "downloaded")
    log(f"done: {len(states)} datasets, {result['total_bytes']:,} bytes verified ({result['downloaded_bytes']:,} downloaded this run)")
    return result


# ------------------------------------------------------------------------------------------------ for the converters
def _pinned_commit(ext, ds) -> Optional[str]:
    """The commit sha recorded by an earlier run of this dataset (None when there is no valid provenance.json)."""
    p = Path(ext) / folder_of(ds["repo"]) / "provenance.json"
    try:
        sha = json.loads(p.read_text(encoding="utf-8")).get("commit_sha")
    except (OSError, ValueError, AttributeError):
        return None
    return sha if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{40}", sha) else None


def _provenance_of(root, ds, ext_dirname: str = EXTERNAL_DIRNAME) -> dict:
    p = Path(root) / ext_dirname / folder_of(ds["repo"]) / "provenance.json"
    if not os.path.isfile(p):
        raise AcquireError(f"{ds['repo']} has not been acquired: {p} is missing")
    return json.loads(p.read_text(encoding="utf-8"))


def train_files(root, which, datasets=None, ext_dirname: str = EXTERNAL_DIRNAME) -> list:
    """Paths of the TRAIN-eligible data files of one dataset (key, repo id or folder name).  dev/validation can never come out of here."""
    table = list(DATASETS if datasets is None else datasets)
    ds = select(table, [which])[0]
    prov = _provenance_of(root, ds, ext_dirname)
    out = []
    for f in prov["files"]:
        if f["role"] == "train" and f["may_train"]:
            if any(part.startswith("do_not_train") for part in PurePosixPath(f["dest_rel"]).parts):
                raise UnsafeLayout(f"{f['dest_rel']} is under a do_not_train folder but marked train")
            out.append(Path(root) / ext_dirname / f["dest_rel"])
    return out


def do_not_train_files(root, datasets=None, ext_dirname: str = EXTERNAL_DIRNAME) -> dict:
    """{repo: [paths]} of the dev/validation files, for decontamination only."""
    table = list(DATASETS if datasets is None else datasets)
    out = {}
    for ds in table:
        p = Path(root) / ext_dirname / folder_of(ds["repo"]) / "provenance.json"
        if not os.path.isfile(p):
            continue
        prov = json.loads(p.read_text(encoding="utf-8"))
        files = [Path(root) / ext_dirname / f["dest_rel"] for f in prov["files"] if f["role"] in NO_TRAIN_ROLES]
        if files:
            out[ds["repo"]] = files
    return out


def load_manifest(root, datasets=None, ext_dirname: str = EXTERNAL_DIRNAME) -> dict:
    """{repo: provenance.json content} for every acquired dataset."""
    table = list(DATASETS if datasets is None else datasets)
    out = {}
    for ds in table:
        p = Path(root) / ext_dirname / folder_of(ds["repo"]) / "provenance.json"
        if os.path.isfile(p):
            out[ds["repo"]] = json.loads(p.read_text(encoding="utf-8"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fetch the six approved Hugging Face datasets into Drive (run on Colab).")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--only", default=None, help="comma separated keys / repo ids")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stage-dir", default=None)
    ap.add_argument("--keep-stage", action="store_true")
    ap.add_argument("--no-verify-after-copy", action="store_true")
    ap.add_argument("--allow-local", action="store_true", help="tests only: the owner's PC must not be loaded")
    ap.add_argument("--now-utc", default=None)
    a = ap.parse_args(argv)
    res = acquire(a.root, a.only, dry_run=a.dry_run, now_utc=a.now_utc, stage_dir=a.stage_dir, keep_stage=a.keep_stage,
                  verify_after_copy=not a.no_verify_after_copy, allow_local=a.allow_local)
    if not a.dry_run:
        brief = {r: {"commit": d["commit_sha"], "provenance": d["provenance"], "files": {x["status"]: sum(1 for y in d["files"] if y["status"] == x["status"]) for x in d["files"]}}
                 for r, d in res["datasets"].items()}
        print(json.dumps({"retrieved_utc": res["retrieved_utc"], "total_bytes": res["total_bytes"], "downloaded_bytes": res["downloaded_bytes"], "datasets": brief}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
