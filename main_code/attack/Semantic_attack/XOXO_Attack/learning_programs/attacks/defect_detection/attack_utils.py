import argparse
import json
import os

import numpy as np
import torch
import transformers
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, SequentialSampler, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from learning_programs.attacks import common
from learning_programs.attacks.attack_utils import (
    get_masked_code_by_position,
    get_identifier_posistions_from_code,
    map_chromesome
)
from learning_programs.attacks.parser_utils import extract_dataflow
from learning_programs.datasets.defect_detection import DATA_PATH


class InputFeatures_GCB(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 position_idx,
                 dfg_to_code,
                 dfg_to_dfg,
                 idx,
                 label

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg
        self.idx=str(idx)
        self.label=label


class InputFeaturesDefault(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label


def convert_code_to_features(code, tokenizer, label, args, inp_idx=0):
    if args.model_name == "microsoft/graphcodebert-base":
        return convert_code_to_features_gcb(code, tokenizer, label, args, inp_idx)
    elif args.model_name == "Salesforce/codet5p-110m-embedding" or args.model_name == "Salesforce/codet5p-220m":
        return convert_code_to_features_codet5(code, tokenizer, label, args, inp_idx)
    elif args.model_name == "microsoft/codebert-base":
        return convert_code_to_features_cb(code, tokenizer, label, args, inp_idx)
    else:
        raise ValueError("Model name not supported.")


def convert_code_to_features_codet5(code, tokenizer, label, args, inp_idx=0):
    code = ' '.join(code.split())
    code_tokens = tokenizer.tokenize(code, padding='max_length', truncation=True, max_length=args.block_size)
    input_ids = tokenizer.encode(code, padding='max_length', truncation=True, max_length=args.block_size)
    return InputFeaturesDefault(code_tokens, input_ids, inp_idx, label)


def convert_code_to_features_raw_cb(code, tokenizer, args):
    code=' '.join(code.split())
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return source_tokens, source_ids


def convert_code_to_features_cb(code, tokenizer, label, args, inp_idx=0):
    source_tokens, source_ids = convert_code_to_features_raw_cb(code, tokenizer, args)
    return InputFeaturesDefault(source_tokens, source_ids, inp_idx, label)


def convert_code_to_features_raw_gcb(code, tokenizer: AutoTokenizer, args):
    code=' '.join(code.split())
    dfg, _, code_tokens = extract_dataflow(code, common.LANGUAGE)
    code_tokens=[tokenizer.tokenize('@ '+x, padding=False, truncation=True, max_length=args.block_size)[1:] if idx!=0 else tokenizer.tokenize(x, padding=False, truncation=True, max_length=args.block_size) for idx,x in enumerate(code_tokens)]
    ori2cur_pos={}
    ori2cur_pos[-1]=(0,0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
    code_tokens=[y for x in code_tokens for y in x]

    code_tokens=code_tokens[:args.block_size+args.data_flow_length-3-min(len(dfg),args.data_flow_length)][:args.block_size-3]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
    dfg = dfg[:args.block_size+args.data_flow_length-len(source_tokens)]
    source_tokens += [x[0] for x in dfg]
    position_idx+=[0 for x in dfg]
    source_ids+=[tokenizer.unk_token_id for x in dfg]
    padding_length=args.block_size+args.data_flow_length-len(source_ids)
    position_idx+=[tokenizer.pad_token_id]*padding_length
    source_ids+=[tokenizer.pad_token_id]*padding_length

    reverse_index={}
    for idx,x in enumerate(dfg):
        reverse_index[x[1]]=idx
    for idx,x in enumerate(dfg):
        dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
    dfg_to_dfg=[x[-1] for x in dfg]
    dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
    length=len([tokenizer.cls_token])
    dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]
    
    return source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg


def convert_code_to_features_gcb(code, tokenizer: AutoTokenizer, label, args, inp_idx=0):
    source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg = convert_code_to_features_raw_gcb(code, tokenizer, args)
    return InputFeatures_GCB(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg, inp_idx, label)


def convert_examples_to_features(js,tokenizer,args):
    return convert_code_to_features(js["func"], tokenizer, int(js['target']), args, int(js['idx']))


def compute_fitness(chromesome, codebert_tgt, tokenizer_tgt, orig_prob, orig_label, true_label, code, names_positions_dict, args):
    temp_code = map_chromesome(chromesome, code, common.LANGUAGE)
    new_feature = convert_code_to_features(temp_code, tokenizer_tgt, true_label, args)
    new_dataset = CodeDataset([new_feature], args)
    new_logits, preds = codebert_tgt.get_results(new_dataset, args.eval_batch_size)
    fitness_value = orig_prob - new_logits[0][orig_label]
    return fitness_value, preds[0]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = 0

    def apply(self, batch):
        inputs = batch[0].to(self.args.device)
        return self.forward(inputs)

    def get_results(self, dataset, batch_size):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size,num_workers=4,pin_memory=False)

        self.eval()
        probs=[]
        labels=[]
        for batch in eval_dataloader:
            label = batch[-1]
            with torch.no_grad():
                prob = self.apply(batch)
                probs.append(prob.cpu().numpy())
                labels.append(label.cpu().numpy())

        probs=np.concatenate(probs,0)
        labels=np.concatenate(labels,0)

        pred_labels = (probs > 0.5).astype(int).tolist()
        probs = [[1 - prob, prob] for prob in probs]
        return probs, pred_labels


class CodeBERTModel(Model):   
    def __init__(self, encoder, args):
        super().__init__()
        self.encoder = encoder
        self.args=args


    def forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        logits=outputs
        prob=F.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob.squeeze(-1)


class GraphCodeBERTModel(Model):
    def __init__(self, encoder, config, args):
        super().__init__()
        self.encoder = encoder
        self.config=config
        self.args=args

    def apply(self, batch):
        inputs = batch[0].to(self.args.device)
        attn_mask = batch[1].to(self.args.device)
        position_idx = batch[2].to(self.args.device)
        return self.forward((inputs, attn_mask, position_idx))
        
    def forward(self, inputs, labels = None):
        inputs_ids, attn_mask, position_idx = inputs
        
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)
    
        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        # https://github.com/microsoft/CodeBERT/issues/73
        outputs = self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx, token_type_ids=position_idx.eq(-1).long())[0]

        logits=outputs
        prob=F.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob.squeeze(-1)


class CodeT5Model(Model):
    def __init__(self, encoder, config, args):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.classifier = torch.nn.Linear(config.embed_dim if args.model_name == "Salesforce/codet5p-110m-embedding" else config.hidden_size, 2)
        self.args = args

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, self.args.block_size)

        if self.args.model_name == "Salesforce/codet5p-110m-embedding":
            vec = self.encoder(input_ids=source_ids)
        else:
            vec = self.get_t5_vec(source_ids)

        logits = self.classifier(vec)
        prob = F.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob[:, 1]


def initialize_target_model(args: argparse.Namespace):
    trust_remote_code = args.model_name == "Salesforce/codet5p-110m-embedding"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=args.model_name)
    print("Loading target model...")
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=trust_remote_code)
    config.num_labels = 1
    transformers.logging.set_verbosity_error()
    if args.model_name == "Salesforce/codet5p-110m-embedding":
        model_class = AutoModel.from_pretrained(args.model_name, config=config, trust_remote_code=trust_remote_code)
    else:
        model_class = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
    transformers.logging.set_verbosity_info()
    if args.model_name == "microsoft/graphcodebert-base":
        model = GraphCodeBERTModel(model_class, config, args)
    elif args.model_name == "microsoft/codebert-base":
        model = CodeBERTModel(model_class, args)
    elif args.model_name == "Salesforce/codet5p-110m-embedding":
        model = CodeT5Model(model_class, config, args)
    else:
        raise ValueError("Model name not supported.")
    model.load_state_dict(torch.load(args.target_weights_path))
    return model.to(args.device).eval(), tokenizer


def get_importance_score(args, example, words_list: list, variable_names: list, tgt_model, tokenizer):
    '''Compute the importance score of each variable'''
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    if len(positions) == 0:
        return None, None, None

    new_example = []

    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)

    for index, tokens in enumerate([words_list] + masked_token_list):
        new_code = ' '.join(tokens)
        new_feature = convert_code_to_features(new_code, tokenizer, example[-1].item(), args)
        new_example.append(new_feature)
    new_dataset = CodeDataset(new_example, args)

    logits, preds = tgt_model.get_results(new_dataset, args.eval_batch_size)
    orig_probs = logits[0]
    orig_label = preds[0]
    orig_prob = max(orig_probs)

    importance_score = []
    for prob in logits[1:]:
        importance_score.append(orig_prob - prob[orig_label])

    return importance_score, replace_token_positions, positions


def load_examples(split, tokenizer, args):
    examples = []
    with open(os.path.join(DATA_PATH, f"{split}.jsonl"), "r") as f:
        for line in tqdm(f):
            js=json.loads(line.strip())
            examples.append(convert_examples_to_features(js,tokenizer,args))
    return examples


def gcb_attn_mask_raw(input_ids: list[int], position_idx: list[int], dfg_to_code: dict, dfg_to_dfg: dict, args: argparse.Namespace) -> torch.Tensor:
    #calculate graph-guided masked function
    attn_mask=torch.zeros((args.block_size+args.data_flow_length,
                    args.block_size+args.data_flow_length),dtype=torch.bool)
    #calculate begin index of node and max length of input
    node_index=sum([i>1 for i in position_idx])
    max_length=sum([i!=1 for i in position_idx])
    #sequence can attend to sequence
    attn_mask[:node_index,:node_index]=True
    #special tokens attend to all tokens
    for idx,i in enumerate(input_ids):
        if i in [0,2]:
            attn_mask[idx,:max_length]=True
    #nodes attend to code tokens that are identified from
    for idx,(a,b) in enumerate(dfg_to_code):
        if a<node_index and b<node_index:
            attn_mask[idx+node_index,a:b]=True
            attn_mask[a:b,idx+node_index]=True
    #nodes attend to adjacent nodes 
    for idx,nodes in enumerate(dfg_to_dfg):
        for a in nodes:
            if a+node_index<len(position_idx):
                attn_mask[idx+node_index,a+node_index]=True
    return attn_mask


def gcb_attn_mask(inp_features: InputFeatures_GCB, args: argparse.Namespace) -> torch.Tensor:
    return gcb_attn_mask_raw(inp_features.input_ids, inp_features.position_idx, inp_features.dfg_to_code, inp_features.dfg_to_dfg, args)


class CodeDataset(Dataset):
    def __init__(self, examples, args):
        self.examples = examples
        self.args = args

    @classmethod
    def from_split(cls, split, tokenizer, args):
        return cls(load_examples(split, tokenizer, args), args)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        if self.args.model_name == "microsoft/graphcodebert-base":
            return (torch.tensor(self.examples[item].input_ids),
                    gcb_attn_mask(self.examples[item], self.args),
                    torch.tensor(self.examples[item].position_idx),
                    torch.tensor(self.examples[item].label))
        return torch.tensor(self.examples[item].input_ids), torch.tensor(self.examples[item].label)


def is_correctly_classified(target: Model, example: tuple[torch.Tensor, torch.Tensor], eval_batch_size: int) -> bool:
    _, orig_label = target.get_results([example], eval_batch_size)
    orig_label = orig_label[0]
    ground_truth = example[-1].item()
    target.query = 0
    return orig_label == ground_truth
