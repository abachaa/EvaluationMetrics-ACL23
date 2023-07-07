from pydoc import doc
import pandas as pd
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str,  help='The output path to save the weight')
parser.add_argument('--candidate', type=str,  help='The reference path')
parser.add_argument('--reference', type=str, help='The candidate path')
parser.add_argument('--source', type=str,  help='The source path')
args = parser.parse_args()



def update_pkl(data, weights, id_text, system_id=None):
    if system_id is None:
        for token_id in weights:
            data[token_id+1][id_text] = weights[token_id]['weight']
    else:
        for token_id in weights:
            data[token_id+1]['sys_summs'][system_id][id_text] = weights[token_id]['weight']
    return  data

def read_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


data_dict2 = {}
i=0
cand_results = args.candidate
ref_results = args.reference
cand_results_weights = args.output + "weight_cands.pkl"
ref_results_weights = args.output + "weight_ref.pkl"

doc_id = 0 
 
with open(cand_results) as f:
    for line in f:
        doc_id = doc_id +1
        if doc_id not in data_dict2:
            data_dict2[doc_id] = {}
        data_dict2[doc_id]['sys_summs'] = {}
        data_dict2[doc_id]['sys_summs']['system'] = {}
        data_dict2[doc_id]['sys_summs']['system']['sys_summ'] = line.strip()
        data_dict2[doc_id]['sys_summs']['system']['scores'] = {}

doc_id = 0 
with open(ref_results) as f:
    for line in f:
        doc_id = doc_id +1
        data_dict2[doc_id]["ref_summ"] = line.strip()
        data_dict2[doc_id]["src"] = "empty"
        data_dict2[doc_id]["id"] = doc_id
        
i = 0
with open(args.source) as f:
    for line in f:
        i = i +1
        data_dict2[i]["src"] = line.strip()

cad = read_pickle(cand_results_weights)
ref = read_pickle(ref_results_weights)      
data_dict2 = update_pkl(data_dict2, ref, "ref_weight")
data_dict2 = update_pkl(data_dict2,cad, "sys_weight", system_id='system')

output = open(args.output + 'full.pkl', 'wb')
pickle.dump(data_dict2, output)
output.close()
