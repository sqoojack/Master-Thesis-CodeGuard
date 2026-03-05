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
from transformers import AutoConfig, AutoTokenizer, AutoModel

from learning_programs.attacks import common
from learning_programs.attacks.attack_utils import (
    get_masked_code_by_position,
    get_identifier_posistions_from_code,
    map_chromesome
)
from learning_programs.attacks.defect_detection.attack_utils import (
    convert_code_to_features_raw_cb,
    convert_code_to_features_raw_gcb,
    gcb_attn_mask_raw
)
from learning_programs.datasets.clone_detection import DATA_PATH


class InputFeatures_GCB(object):
    """A single training/test features for a example."""
    def __init__(self,
             input_tokens_1,
             input_ids_1,
             position_idx_1,
             dfg_to_code_1,
             dfg_to_dfg_1,
             input_tokens_2,
             input_ids_2,
             position_idx_2,
             dfg_to_code_2,
             dfg_to_dfg_2,
             label,
             url1,
             url2

    ):
        #The first code function
        self.input_tokens_1 = input_tokens_1
        self.input_ids_1 = input_ids_1
        self.position_idx_1=position_idx_1
        self.dfg_to_code_1=dfg_to_code_1
        self.dfg_to_dfg_1=dfg_to_dfg_1
        
        #The second code function
        self.input_tokens_2 = input_tokens_2
        self.input_ids_2 = input_ids_2
        self.position_idx_2=position_idx_2
        self.dfg_to_code_2=dfg_to_code_2
        self.dfg_to_dfg_2=dfg_to_dfg_2
        
        #label
        self.label=label
        self.url1=url1
        self.url2=url2


class InputFeatures_CB(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,
                 url1,
                 url2

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label
        self.url1=url1
        self.url2=url2


def convert_code_to_features(code1, code2, tokenizer, label, args, inp_idx_1=0, inp_idx_2=1, cache=None):
    if args.model_name == "microsoft/graphcodebert-base":
        return convert_code_to_features_gcb(code1, code2, tokenizer, label, args, inp_idx_1, inp_idx_2, cache)
    else:
        return convert_code_to_features_cb(code1, code2, tokenizer, label, args, inp_idx_1, inp_idx_2, cache)


def convert_code_to_features_cb(code1, code2, tokenizer, label, args, inp_idx_1=0, inp_idx_2=1, cache=None):
    if cache is None:
        cache = dict()
    if inp_idx_1 not in cache:
        code1_tokens, code1_ids = convert_code_to_features_raw_cb(code1, tokenizer, args)
        cache[inp_idx_1] = (code1_tokens, code1_ids)
    code1_tokens, code1_ids = cache[inp_idx_1]
    if inp_idx_2 not in cache:
        code2=' '.join(code2.split())
        code2_tokens, code2_ids = convert_code_to_features_raw_cb(code2, tokenizer, args)
        cache[inp_idx_2] = (code2_tokens, code2_ids)
    code2_tokens, code2_ids = cache[inp_idx_2]
    source_tokens=code1_tokens+code2_tokens
    source_ids=code1_ids+code2_ids
    return InputFeatures_CB(source_tokens, source_ids, label, inp_idx_1, inp_idx_2)


def convert_code_to_features_gcb(code1, code2, tokenizer: AutoTokenizer, label, args, inp_idx_1=0, inp_idx_2=1, cache=None):
    if cache is None:
        cache = dict()
    if inp_idx_1 not in cache:
        source_tokens_1, source_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1 = convert_code_to_features_raw_gcb(code1, tokenizer, args)
        cache[inp_idx_1] = (source_tokens_1, source_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1)
    if inp_idx_2 not in cache:
        source_tokens_2, source_ids_2, position_idx_2, dfg_to_code_2, dfg_to_dfg_2 = convert_code_to_features_raw_gcb(code2, tokenizer, args)
        cache[inp_idx_2] = (source_tokens_2, source_ids_2, position_idx_2, dfg_to_code_2, dfg_to_dfg_2)
    source_tokens_1, source_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1 = cache[inp_idx_1]
    source_tokens_2, source_ids_2, position_idx_2, dfg_to_code_2, dfg_to_dfg_2 = cache[inp_idx_2]
    return InputFeatures_GCB(source_tokens_1, source_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1, source_tokens_2, source_ids_2, position_idx_2, dfg_to_code_2, dfg_to_dfg_2, label, inp_idx_1, inp_idx_2)


def compute_fitness(chromesome, code_2, codebert_tgt, tokenizer_tgt, orig_prob, orig_label, true_label , code_1, names_positions_dict, args):
    temp_replace = map_chromesome(chromesome, code_1, common.LANGUAGE)
    new_feature = convert_code_to_features(temp_replace, code_2, tokenizer_tgt, true_label, args)
    new_dataset = CodeDataset([new_feature], args)
    new_logits, preds = codebert_tgt.get_results(new_dataset, args.eval_batch_size)
    fitness_value = orig_prob - new_logits[0][orig_label]
    return fitness_value, preds[0]


class RobertaClassificationHeadWithDropout(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, dim):
        super().__init__()
        self.dense = nn.Linear(dim * 2, dim)
        self.out_proj = nn.Linear(dim, 2)

    def forward(self, x, **kwargs):
        x = x.reshape(-1, x.size(-1) * 2)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


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
    def __init__(self, encoder,config,args):
        super().__init__()
        self.encoder = encoder
        self.config=config
        self.classifier=RobertaClassificationHeadWithDropout(config)
        self.args=args
    
        
    def forward(self, input_ids=None,labels=None): 
        input_ids=input_ids.view(-1,self.args.block_size)
        outputs = self.encoder(input_ids= input_ids,attention_mask=input_ids.ne(1))[0]
        logits=self.classifier(outputs)
        prob=F.softmax(logits, dim=-1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob[:, 1]


class GraphCodeBERTModel(Model):
    def __init__(self, encoder,config,args):
        super().__init__()
        self.encoder = encoder
        self.config=config
        self.classifier=RobertaClassificationHeadWithDropout(config)
        self.args=args

    def apply(self, batch):
        return self.forward(
            (batch[0].to(self.args.device),
            batch[1].to(self.args.device),
            batch[2].to(self.args.device),
            batch[3].to(self.args.device),
            batch[4].to(self.args.device),
            batch[5].to(self.args.device),
            )
        )

    def forward(self, inputs,labels=None):
        inputs_ids_1,position_idx_1,attn_mask_1,inputs_ids_2,position_idx_2,attn_mask_2 = inputs

        bs,l=inputs_ids_1.size()
        inputs_ids=torch.cat((inputs_ids_1.unsqueeze(1),inputs_ids_2.unsqueeze(1)),1).view(bs*2,l)
        position_idx=torch.cat((position_idx_1.unsqueeze(1),position_idx_2.unsqueeze(1)),1).view(bs*2,l)
        attn_mask=torch.cat((attn_mask_1.unsqueeze(1),attn_mask_2.unsqueeze(1)),1).view(bs*2,l,l)

        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]

        # https://github.com/microsoft/CodeBERT/issues/73
        outputs = self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx, token_type_ids=position_idx.eq(-1).long())[0]

        logits=self.classifier(outputs)
        prob=F.softmax(logits, dim=-1)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob[:, 1]


class CodeT5Model(Model):
    def __init__(self, encoder, config, args):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.classifier = RobertaClassificationHead(config.embed_dim if args.model_name == "Salesforce/codet5p-110m-embedding" else config.hidden_size)
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

    def get_bart_vec(self, source_ids):
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

    def get_roberta_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        vec = self.encoder(input_ids=source_ids, attention_mask=attention_mask)[0][:, 0, :]
        return vec

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, self.args.block_size)

        if self.args.model_name == "Salesforce/codet5p-110m-embedding":
            vec = self.encoder(input_ids=source_ids)
        elif self.args.model_type == 'codet5':
            vec = self.get_t5_vec(source_ids)
        elif self.args.model_type == 'bart':
            vec = self.get_bart_vec(source_ids)
        elif self.args.model_type == 'roberta':
            vec = self.get_roberta_vec(source_ids)

        logits = self.classifier(vec)
        prob = F.softmax(logits, dim=-1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob[:, 1]


def initialize_target_model(args: argparse.Namespace):
    trust_remote_code = args.model_name == "Salesforce/codet5p-110m-embedding"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=trust_remote_code)
    print("Loading target model...")
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=trust_remote_code)
    config.num_labels = 2
    transformers.logging.set_verbosity_error()
    model_class = AutoModel.from_pretrained(args.model_name, config=config, trust_remote_code=trust_remote_code)
    transformers.logging.set_verbosity_info()
    if args.model_name == "microsoft/graphcodebert-base":
        model = GraphCodeBERTModel(model_class, config, args)
    elif args.model_name == "microsoft/codebert-base":
        model = CodeBERTModel(model_class, config, args)
    elif args.model_name == "Salesforce/codet5p-110m-embedding":
        model = CodeT5Model(model_class, config, args)
    else:
        raise ValueError("Model name not supported.")
    model.load_state_dict(torch.load(args.target_weights_path))
    return model.to(args.device).eval(), tokenizer


def get_importance_score(args, example, code_2, words_list: list, variable_names: list, tgt_model, tokenizer):
    '''Compute the importance score of each variable'''
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    if len(positions) == 0:
        return None, None, None

    new_example = []

    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)

    for code1_tokens in [words_list] + masked_token_list:
        code1 = ' '.join(code1_tokens)
        new_feature = convert_code_to_features(code1,code_2,tokenizer,example[-1].item(), args)
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
    index_filename=os.path.join(DATA_PATH, f"{split}_processed.txt")

    with open('/'.join(index_filename.split('/')[:-1])+'/functions.json') as f:
        url_to_code = json.load(f)
                    
    #load code function according to index
    data=[]
    cache={}
    with open(index_filename) as f:
        for line in f:
            line=line.strip()
            _, url1,url2,label=line.strip().split()
            label = 0 if label == '0' else 1
            code1 = url_to_code[url1]
            code2 = url_to_code[url2]
            data.append((code1, code2, tokenizer, label, args, url1, url2, cache))

    return [convert_code_to_features(*x) for x in tqdm(data,total=len(data))]


def gcb_attn_mask(inp_features: InputFeatures_GCB, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    attn_mask_1 = gcb_attn_mask_raw(inp_features.input_ids_1, inp_features.position_idx_1, inp_features.dfg_to_code_1, inp_features.dfg_to_dfg_1, args)
    attn_mask_2 = gcb_attn_mask_raw(inp_features.input_ids_2, inp_features.position_idx_2, inp_features.dfg_to_code_2, inp_features.dfg_to_dfg_2, args)
    return attn_mask_1, attn_mask_2


class CodeDataset(Dataset):
    def __init__(self, examples, args) -> None:
        self.examples = examples
        self.args = args

    @classmethod
    def from_split(cls, split, tokenizer, args):
        return cls(load_examples(split, tokenizer, args), args)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        if self.args.model_name == "microsoft/graphcodebert-base":
            attn_mask_1, attn_mask_2 = gcb_attn_mask(self.examples[item], self.args)
            return (torch.tensor(self.examples[item].input_ids_1),
                    torch.tensor(self.examples[item].position_idx_1),
                    attn_mask_1,
                    torch.tensor(self.examples[item].input_ids_2),
                    torch.tensor(self.examples[item].position_idx_2),
                    attn_mask_2,
                    torch.tensor(self.examples[item].label))
        return torch.tensor(self.examples[item].input_ids),torch.tensor(self.examples[item].label) 


def is_correctly_classified(target: Model, example: tuple[torch.Tensor, torch.Tensor] , eval_batch_size: int) -> bool:
    _, orig_label = target.get_results([example], eval_batch_size)
    orig_label = orig_label[0]
    ground_truth = example[-1].item()
    target.query = 0
    return orig_label == ground_truth
