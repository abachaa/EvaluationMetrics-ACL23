import torch

import sys
import argparse
import os
from re import A
import time
import pickle
import numpy as np
from medcat.cat import CAT
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


tui_id = {'T017': 5, 'T029': 6, 'T023': 7, 'T030': 8, 'T031': 9, 'T022': 10, 'T025': 11, 'T026': 12, 'T018': 13,
          'T021': 14, 'T024': 15, 'T116': 16, 'T195': 17, 'T123': 18, 'T122': 19, 'T103': 20, 'T120': 21,
          'T104': 22, 'T200': 23, 'T196': 24, 'T126': 25, 'T131': 26, 'T125': 27, 'T129': 28, 'T130': 29,
          'T197': 30, 'T114': 31, 'T109': 32, 'T121': 33, 'T192': 34, 'T127': 35, 'T190': 36, 'T049': 37,
          'T019': 38, 'T047': 39, 'T050': 40, 'T061': 41, 'T037': 42, 'T048': 43, 'T191': 44, 'T046': 45,
          'T184': 46, 'T020': 47, 'T060': 48, 'T065': 49, 'T058': 50, 'T059': 51, 'T063': 52, 'T062': 53}



class Medical_Weight:

    def __init__(self, file_path, device='cuda:0',  max_length=1024, checkpoint='microsoft/deberta-xlarge-mnli'):

        self.device = device
        MODEL_DIR = file_path
        model_pack_path = MODEL_DIR + "umls_sm_wstatus_2021_oct.zip"
        self.cat = CAT.load_model_pack(model_pack_path)

        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, return_offsets_mapping=True)


    def data_iterator_test(self, token_list):
        i = -1
        for text_id in token_list:
            i = i + 1
            yield (text_id, token_list[text_id]['text'])

    def medical_tokenizer(self,  srcs,   batch_size=1):
        """ Score a batch of examples """
        final_encoding = {}
        i =0
        j = 0
        for doc in range(0, len(srcs)):
 
            offset_mapping = self.tokenizer(srcs[doc],return_offsets_mapping=True, add_special_tokens=True)['offset_mapping']
            desired_output = []
            for token in offset_mapping:
                desired_output.append(token)

            final_encoding[doc] = {}
            final_encoding[doc]['text'] = srcs[doc]
            final_encoding[doc]['tokens'] =desired_output

        return final_encoding

    def create_bartweight(self, docs, results):
        for doc in tqdm(list(results.keys())):
            entity_id = {}
            weigths_tui = []
            text = docs[doc]['text']
            tokens = docs[doc]['tokens']

            test_dokimi = []
            for annotation in list(results[doc]['entities'].values()):
                if len(annotation['type_ids']) > 0:  
                    if annotation['type_ids'][0] in tui_id:                      
                        start = annotation['start']  
                        end_entity = annotation['end'] 
                        if start not in entity_id:
                            entity_id[start-1] = end_entity 
                        else:
                            if end_entity >entity_id[start-1]:
                                entity_id[start-1]= end_entity 
                        test_dokimi.append(text[start:end_entity])
            entity_flag = False
            entity_end = -1
            for token in tokens:
                if token[0] in entity_id:  
                    weigths_tui.append(2)
                    entity_end = entity_id[token[0]] 
                    entity_flag = True
                
                else:
                    if entity_flag and token[1] <=entity_end:
                        weigths_tui.append(2)
                    else:
                        entity_flag = False
                        weigths_tui.append(1)
            docs[doc]['weight'] = weigths_tui
         
            
        return docs



    def get_weight(self, candidate_path, reference_path, output_path):

        with open(candidate_path) as f:
            cands = [line.strip() for line in f]
     
  
        token_list = self.medical_tokenizer(cands, batch_size=4)

        batch_size_chars = 500000  

        # Run model
        results = self.cat.multiprocessing(self.data_iterator_test(token_list), batch_size_chars=batch_size_chars, nproc=4)  # Number of processors

        weigths = self.create_bartweight(token_list, results)
        output = open(output_path + "test_weight_cands.pkl", "wb")
        pickle.dump( weigths, output)

        with open(reference_path) as f:
            refs = [line.strip() for line in f]
       

        token_list  = self.medical_tokenizer(refs, batch_size=4)

        batch_size_chars = 500000 
        results = self.cat.multiprocessing(self.data_iterator_test(token_list), batch_size_chars=batch_size_chars,
                                           nproc=4) 

        weigths = self.create_bartweight(token_list, results)
        output = open(output_path + "test_weight_ref.pkl", "wb")
        pickle.dump(weigths, output)
        # ================================================================================================


parser = argparse.ArgumentParser(description='Scorer parameters')
parser.add_argument('--file', type=str,  help='Medcat file path')
parser.add_argument('--device', type=str, default='cuda:0',  help='The device to run on.')
parser.add_argument('--output', type=str, help='The output path to save the calculated scores.')
parser.add_argument('--candidate', type=str,  help='The candidate path')
parser.add_argument('--refecrence', type=str, help='The reference path')

start_time = time.time()
args = parser.parse_args()
weight = Medical_Weight(args.file,  args.device)
weight.get_weight(args.candidate, args.reference,args.output )

