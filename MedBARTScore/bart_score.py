# %%
import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import GPT2Tokenizer, OPTForCausalLM, GPT2LMHeadModel, GPTJForCausalLM
from tqdm import tqdm

class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)


        self.max_length  = 1000
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        self.model.to(device)
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self , path ):
        """ Load model from paraphrase finetuning """
        self.model.load_state_dict(torch.load(path , map_location=self.device))

   
    def score(self, srcs, tgts,  src_weight=None, tgt_weight=None, batch_size=4,promtp=0):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            tgt_weight_list =  None
            if    (tgt_weight is not None):
                #src_weight_list = src_weight[i: i + batch_size]
                tgt_weight_list = tgt_weight[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)

                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                     #======================================================
                 
                    if tgt_weight_list is not None:
                        weight = torch.stack(tgt_weight_list)[:, :loss.size()[1]].to(self.device)
                        loss = torch.mul(weight, loss) 
                        loss1 = loss.sum(dim=1) / tgt_len

                    else:
                        loss1 = loss.sum(dim=1) / tgt_len

                    try:
                        curr_score_list = [-x.item() for x in loss1]
                    except:
                        print(loss)
                        exit(5)
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list
