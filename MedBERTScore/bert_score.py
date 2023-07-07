from hashlib import new
import os
import pathlib
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from transformers import AutoTokenizer
import pickle
from transformers import RobertaForSequenceClassification, RobertaModel

from utils import (bert_cos_score_idf, cache_scibert, get_bert_embedding,
                    get_hash, get_idf_dict, get_model, get_tokenizer, 
                    lang2model, model2layers, sent_encode)

__all__ = ["score"]

def read_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data

def update_idf(old_dict, new_dict= {}):

    for id in old_dict:
        new_dict[old_dict[id]['text']] =  old_dict[id]['weight']
    return new_dict

def score(
    cands,
    refs,
    model_type=None,
    num_layers=None,
    verbose=False,
    idf=False,
    device=None,
    batch_size=2,
    nthreads=4,
    all_layers=False,
    lang=None,
    return_hash=False,
    rescale_with_baseline=False,
    baseline_path=None,
    use_fast_tokenizer=False,
    semantic_dictionary = None,
    weights_per_row_cands = None,
    weights_per_row_ref = None,sliding_window_flag=False
):
    """
    BERTScore metric.

    Args:
        - :param: `cands` (list of str): candidate sentences
        - :param: `refs` (list of str or list of list of str): reference sentences
        - :param: `model_type` (str): bert specification, default using the suggested
                  model for the target langauge; has to specify at least one of
                  `model_type` or `lang`
        - :param: `num_layers` (int): the layer of representation to use.
                  default using the number of layer tuned on WMT16 correlation data
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `idf` (bool or dict): use idf weighting, can also be a precomputed idf_dict
        - :param: `device` (str): on which the contextual embedding model will be allocated on.
                  If this argument is None, the model lives on cuda:0 if cuda is available.
        - :param: `nthreads` (int): number of threads
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `lang` (str): language of the sentences; has to specify
                  at least one of `model_type` or `lang`. `lang` needs to be
                  specified when `rescale_with_baseline` is True.
        - :param: `return_hash` (bool): return hash code of the setting
        - :param: `rescale_with_baseline` (bool): rescale bertscore with pre-computed baseline
        - :param: `baseline_path` (str): customized baseline file
        - :param: `use_fast_tokenizer` (bool): `use_fast` parameter passed to HF tokenizer

    Return:
        - :param: `(P, R, F)`: each is of shape (N); N = number of input
                  candidate reference pairs. if returning hashcode, the
                  output will be ((P, R, F), hashcode). If a candidate have
                  multiple references, the returned score of this candidate is
                  the *best* score among all references.
    """
    assert len(cands) == len(refs), "Different number of candidates and references"

    assert (
        lang is not None or model_type is not None
    ), "Either lang or model_type should be specified"

    ref_group_boundaries = None
    if not isinstance(refs[0], str):
        print(refs[0])
        ref_group_boundaries = []
        ori_cands, ori_refs = cands, refs
        cands, refs = [], []
        count = 0
        for cand, ref_group in zip(ori_cands, ori_refs):
            cands += [cand] * len(ref_group)
            if len(ref_group) > 1:
                print(ref_group)
            refs += ref_group
            ref_group_boundaries.append((count, count + len(ref_group)))
            count += len(ref_group)

    if rescale_with_baseline:
        assert lang is not None, "Need to specify Language when rescaling with baseline"

    if model_type is None:
        lang = lang.lower()
        model_type = lang2model[lang]
    if num_layers is None:
        num_layers = model2layers[model_type]
    
    
    model_type = "microsoft/deberta-xlarge-mnli"
    tokenizer = get_tokenizer(model_type, use_fast_tokenizer)
    cat  = None 
    
    model =  get_model(model_type, num_layers, all_layers)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
   
    idf_per_text = None
    idf_dict = None

    if semantic_dictionary is not None:
         
        idf_dict =  read_pickle(semantic_dictionary)
        idf_dict = defaultdict(lambda: 1.0, idf_dict)
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0

    elif (weights_per_row_cands is not None) and  (weights_per_row_ref is not None):
         
        idf_per_row_cands = read_pickle(weights_per_row_cands)
        idf_per_row_ref = read_pickle(weights_per_row_ref)
        idf_per_text=  update_idf(idf_per_row_cands)
        idf_per_text=  update_idf(idf_per_row_ref, idf_per_text)
        
    else:
        if not idf:
            #print("no dictionary...")
            idf_dict = defaultdict(lambda: 1.0)
            # set idf for [SEP] and [CLS] to 0
            
            idf_dict[tokenizer.sep_token_id] = 0
            idf_dict[tokenizer.cls_token_id] = 0
        elif isinstance(idf, dict):
            if verbose:
                print("using predefined IDF dict...")
            idf_dict = idf
            # set idf for [SEP] and [CLS] to 0
            idf_dict[tokenizer.sep_token_id] = 0
            idf_dict[tokenizer.cls_token_id] = 0

        else:
            if verbose:
                print("preparing IDF dict...")
            start = time.perf_counter()  
            idf_dict = get_idf_dict(refs, tokenizer, nthreads=nthreads, sliding_window_flag=sliding_window_flag)
            if verbose:
                print("done in {:.2f} seconds".format(time.perf_counter() - start))

    if verbose:
        print("calculating scores...")
    start = time.perf_counter()
    all_preds = bert_cos_score_idf(
        model,
        refs,
        cands,
        tokenizer,
        idf_dict,
        idf_per_text,
        cat,
        verbose=verbose,
        device=device,
        batch_size=batch_size,
        all_layers=all_layers,sliding_window_flag=sliding_window_flag
    ).cpu()
    
    if ref_group_boundaries is not None:
        max_preds = []
        for beg, end in ref_group_boundaries:
            max_preds.append(all_preds[beg:end].max(dim=0)[0])
        all_preds = torch.stack(max_preds, dim=0)

    use_custom_baseline = baseline_path is not None
    if rescale_with_baseline:
        if baseline_path is None:
            baseline_path = os.path.join(
                os.path.dirname(__file__), f"rescale_baseline/{lang}/{model_type}.tsv"
            )
        if os.path.isfile(baseline_path):
            if not all_layers:
                baselines = torch.from_numpy(
                    pd.read_csv(baseline_path).iloc[num_layers].to_numpy()
                )[1:].float()
            else:
                baselines = (
                    torch.from_numpy(pd.read_csv(baseline_path).to_numpy())[:, 1:]
                    .unsqueeze(1)
                    .float()
                )

            all_preds = (all_preds - baselines) / (1 - baselines)
        else:
            print(
                f"Warning: Baseline not Found for {model_type} on {lang} at {baseline_path}",
                file=sys.stderr,
            )

    out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F

    if verbose:
        time_diff = time.perf_counter() - start
        print(
            f"done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec"
        )

    if return_hash:
        return tuple(
            [
                out,
                get_hash(
                    model_type,
                    num_layers,
                    idf,
                    rescale_with_baseline,
                    use_custom_baseline=use_custom_baseline,
                    use_fast_tokenizer=use_fast_tokenizer,
                ),
            ]
        )

    return out
