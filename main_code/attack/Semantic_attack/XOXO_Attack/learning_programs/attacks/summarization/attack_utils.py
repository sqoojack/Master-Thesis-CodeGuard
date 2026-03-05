import argparse
from typing import Callable

import numpy as np
import torch
import transformers
from torch import nn
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM

from learning_programs.attacks import common
from learning_programs.attacks.attack_utils import (
    get_masked_code_by_position,
    get_identifier_posistions_from_code,
    map_chromesome
)
from learning_programs.metrics.bleu import compute_bleus


def compute_fitness(chromesome, codebert_tgt, tokenizer_tgt, orig_bleu, nl, code, args, queries_so_far=0):
    temp_code = map_chromesome(chromesome, code, common.LANGUAGE)
    bleus, _, queries = get_results([temp_code], nl, codebert_tgt, tokenizer_tgt, args)
    bleu = bleus[0]
    return orig_bleu - bleu, bleu, queries + queries_so_far


class Seq2SeqCodeBERT(nn.Module):
    def __init__(self, encoder, decoder, config, tokenizer):
        super().__init__()
        self.encoder = encoder
        self.decoder=decoder
        self.config=config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()
        
        self.sos_id=tokenizer.cls_token_id
        self.eos_id=tokenizer.sep_token_id
        
    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight
                  
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)        
        
    def forward(self, input_ids, attention_mask, beam_size, max_length):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_output = outputs[0].permute([1,0,2]).contiguous()
        preds=[]
        zero=torch.cuda.LongTensor(1).fill_(0)
        for i in range(input_ids.shape[0]):
            context=encoder_output[:,i:i+1]
            context_mask=attention_mask[i:i+1,:]
            beam = BeamCodeBERT(beam_size,self.sos_id,self.eos_id)
            input_ids=beam.getCurrentState()
            context=context.repeat(1, beam_size,1)
            context_mask=context_mask.repeat(beam_size,1)
            for _ in range(max_length):
                if beam.done():
                    break
                attn_mask=-1e4 *(1-self.bias[:input_ids.shape[1],:input_ids.shape[1]])
                tgt_embeddings = self.encoder.embeddings(input_ids).permute([1,0,2]).contiguous()
                out = self.decoder(tgt_embeddings,context,tgt_mask=attn_mask,memory_key_padding_mask=(1-context_mask).bool())
                out = torch.tanh(self.dense(out))
                hidden_states=out.permute([1,0,2]).contiguous()[:,-1,:]
                out = self.lsm(self.lm_head(hidden_states)).data
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids=torch.cat((input_ids,beam.getCurrentState()),-1)
            hyp= beam.getHyp(beam.getFinal())
            pred=beam.buildTargetTokens(hyp)[:beam_size]
            pred=[torch.cat([x.view(-1) for x in p]+[zero]*(max_length-len(p))).view(1,-1) for p in pred]
            preds.append(torch.cat(pred,0).unsqueeze(0))
                
        preds=torch.cat(preds,0)

        return preds

    # So that we can use this model interchangeably with AutoModelForSeq2SeqLM
    def generate(self, input_ids, attention_mask, beam_size: int, max_length: int):
        return self.forward(input_ids, attention_mask, beam_size, max_length)


class BeamCodeBERT(object):
    def __init__(self, size,sos,eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                       .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))


        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence


def _batch(iterable, n=1):
    len_ = len(iterable)
    for ndx in range(0, len_, n):
        yield iterable[ndx : min(ndx + n, len_)]


def get_results(codes, nl, model, tokenizer, args, queries_so_far=0) -> tuple[list[float], list[str], int]:
    generations = []
    for batch in _batch(codes, args.eval_batch_size):
        tokenized = tokenizer(batch, padding="max_length", truncation=True, return_tensors="pt", max_length=args.max_source_length).to(args.device)
        outputs = model.generate(**tokenized, max_length=args.max_target_length, num_beams=args.num_beams)
        generations.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return compute_bleus(generations, [nl] * len(generations)), generations, len(generations) + queries_so_far


def initialize_target_model(args: argparse.Namespace) -> tuple[Seq2SeqCodeBERT | AutoModelForSeq2SeqLM, AutoTokenizer]:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    print("Loading target model...")
    transformers.logging.set_verbosity_error()
    if args.model_name == "microsoft/codebert-base":
        config = AutoConfig.from_pretrained(args.model_name)
        encoder_class = AutoModel.from_pretrained(args.model_name, config=config)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = Seq2SeqCodeBERT(encoder_class, decoder, config, tokenizer)
        model.load_state_dict(torch.load(args.target_weights_path))
    elif args.model_name in ["Salesforce/codet5p-220m", "uclanlp/plbart-python-en_XX"]:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.target_weights_path)
    else:
        raise ValueError("Model name not supported.")
    return model.to(args.device).eval(), tokenizer


def get_importance_score(example, words_list: list, variable_names: list, tgt_model, tokenizer, args, queries_so_far=0):
    '''Compute the importance score of each variable'''
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    if len(positions) == 0:
        return None, None, None, None

    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)

    new_codes = [" ".join(tokens) for tokens in [words_list] + masked_token_list]
    bleus, _, queries = get_results(new_codes, example.docstring, tgt_model, tokenizer, args, queries_so_far)

    orig_bleu = bleus[0]
    importance_score = [orig_bleu - bleu for bleu in bleus[1:]]

    return importance_score, replace_token_positions, positions, queries + queries_so_far


BLEU_ZERO_ATOL = 1e-5


def get_criterion(args: argparse.Namespace) -> Callable[[float, float], bool]:
    if args.success_criterion == "zero":
        def is_success(bleu: float, orig_bleu: float) -> bool:
            return np.isclose(bleu, 0.0, atol=BLEU_ZERO_ATOL)
        return is_success

    if args.success_criterion == "abs_threshold":
        def is_success(bleu: float, orig_bleu: float) -> bool:
            return bleu < orig_bleu - args.bleu_threshold
        return is_success

    if args.success_criterion == "rel_threshold":
        def is_success(bleu: float, orig_bleu: float) -> bool:
            return bleu < orig_bleu * (1 - args.bleu_threshold)
        return is_success

    if args.success_criterion == "decrease":
        def is_success(bleu: float, orig_bleu: float) -> bool:
            return bleu < orig_bleu
        return is_success

    raise ValueError(f"Invalid criterion: {args.success_criterion}")
