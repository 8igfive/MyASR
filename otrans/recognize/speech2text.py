import logging
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from otrans.data import EOS, BOS, PAD
from otrans.recognize.base import Recognizer
from tools.soft_dtw import SoftDTW
from packaging import version
from collections import defaultdict
import pdb

logger = logging.getLogger(__name__)

class SpeechToTextRecognizer(Recognizer):
    def __init__(self, model, lm=None, lm_weight=0.1, ctc_weight=0.0, beam_width=5, nbest=1,
                 max_len=50, idx2unit=None, penalty=0, lamda=5, ngpu=1, apply_cache=False, mode='beam'):
        super(SpeechToTextRecognizer, self).__init__(model, idx2unit, lm, lm_weight, ngpu)

        self.beam_width = beam_width
        self.max_len = max_len
        self.nbest = nbest

        self.penalty = penalty
        self.lamda = lamda

        self.ctc_weight = ctc_weight
        self.lm_weight = lm_weight

        self.attn_weights = {}
        self.apply_cache = False 

        self.mode = mode

    def encode(self, inputs, inputs_mask, cache=None):
        new_cache = {}
        inputs, inputs_mask, fe_cache = self.model.frontend.inference(inputs, inputs_mask, cache['frontend'] if cache is not None else None)
        new_cache['frontend'] = fe_cache

        # 新版源码中这里直接用了 forward，他们之间的差别主要在于是否有 drop_out
        memory, memory_mask, enc_cache, enc_attn_weights = self.model.encoder.inference(inputs, inputs_mask, cache['encoder'] if cache is not None else None)
        new_cache['encoder'] = enc_cache

        return memory, memory_mask, new_cache, enc_attn_weights

    def decode(self, preds, memory, memory_mask, cache=None):
        log_probs, dec_cache, dec_attn_weights = self.model.decoder.inference(preds, memory, memory_mask, cache)
        return log_probs, dec_cache, dec_attn_weights

    def _beam(self, inputs, inputs_mask):
        cache = {'fronend': None, 'encoder': None, 'decoder': None, 'lm': None}

        self.attn_weights = {}
        memory, memory_mask, _, enc_attn_weights = self.encode(inputs, inputs_mask)
        
        self.attn_weights['encoder'] = enc_attn_weights
        self.attn_weights['decoder'] = []

        b, t, v = memory.size()
        #print(b)
        #print(t)
        #print(v)
        beam_memory = memory.unsqueeze(1).repeat([1, self.beam_width, 1, 1]).view(b * self.beam_width, t, v)
        beam_memory_mask = memory_mask.unsqueeze(1).repeat([1, self.beam_width, 1]).view(b * self.beam_width, t)
        #print(self.beam_width)
        preds = torch.ones([b * self.beam_width, 1], dtype=torch.long, device=memory.device) * BOS

        scores = torch.FloatTensor([0.0] + [-float('inf')] * (self.beam_width - 1))
        scores = scores.to(memory.device).repeat([b]).unsqueeze(1)
        ending_flag = torch.zeros_like(scores, dtype=torch.bool)


        with torch.no_grad():
            for _ in range(1, self.max_len+1):
                preds, cache, scores, ending_flag = self.decode_step(
                    preds, beam_memory, beam_memory_mask, cache, scores, ending_flag)
                #pdb.set_trace()
                #print(preds)
                #print(beam_memory.shape) #3 94 256
                # whether stop or not
                '''
                criterion = SoftDTW(gamma=1.0, normalize=True) # just like nn.MSELoss()
                
                
                x = beam_memory[:, :min(self.beam_width, self.nbest), 1:]
                y = torch.rand(1,110,255)
                loss = criterion(x, y)
                print(loss)
                '''
                if ending_flag.sum() == b * self.beam_width:
                    break
            #print(self.beam_width)
            scores = scores.view(b, self.beam_width)
            preds = preds.view(b, self.beam_width, -1)

            lengths = torch.sum(torch.ne(preds, EOS).float(), dim=-1)

            # length penalty
            if self.penalty:
                lp = torch.pow((self.lamda + lengths) /
                               (self.lamda + 1), self.penalty)
                scores /= lp

            sorted_scores, offset_indices = torch.sort(scores, dim=-1, descending=True)

            base_indices = torch.arange(b, dtype=torch.long, device=offset_indices.get_device()) * self.beam_width
            base_indices = torch.arange(b, dtype=torch.long, device=offset_indices.device) * self.beam_width
            base_indices = base_indices.unsqueeze(1).repeat([1, self.beam_width]).view(-1) # 由于数值 slice 只在第一层取，所以需要此 base_indices 作为基。
            #print(preds)
            preds = preds.view(b * self.beam_width, -1)
            indices = offset_indices.view(-1) + base_indices
            #print(preds)
            #print('indices: ',indices)
            # remove BOS
            sorted_preds = preds[indices].view(b, self.beam_width, -1)
            #print(sorted_preds)
            nbest_preds = sorted_preds[:, :min(self.beam_width, self.nbest), 1:]
            nbest_scores = sorted_scores[:, :min(self.beam_width, self.nbest)]
            #torch.set_printoptions(profile="full")
            #print(nbest_preds)
            #print(nbest_scores)
        return nbest_preds, nbest_scores

    def _ctc_greedy(self, inputs, inputs_mask):

        if self.ctc_weight == 0:
            logger.error('Lack of CTC module')
            return None, None

        memory, memory_mask, _, enc_attn_weights = self.encode(inputs, inputs_mask)
        
        batch_size, max_len, hidden_size = memory.size()
        
        ctc_probs, memory_lengths = self.model.assistor.inference(memory, memory_mask) # (B, S, V), (B, S)
        pad_mask = self.make_pad_mask(memory_lengths, max_len)
        top1_probs, top1_index = ctc_probs.topk(k=1, dim=-1)                           # (B, S, k), (B, S, k)
        top1_index = top1_index.squeeze(dim=-1).masked_fill_(pad_mask, EOS)            
        nbest_result = top1_index.unsqueeze(dim=1)                                     # (B, beam, S)
        new_nbest_result = torch.ones_like(nbest_result, device=memory.device) * EOS
        
        for i in range(nbest_result.shape[0]):
            new_nbest_result[i][0] = self.remove_dup_and_blank(nbest_result[i][0])
        
        nbest_score = top1_probs.max(dim=1)[0]                                         # (B, beam)
        
        return new_nbest_result, nbest_score

    def _ctc_beam(self, inputs, inputs_mask, require_encoder_out=False, keep_all_result=False):
        if self.ctc_weight == 0:
            logger.error('Lack of CTC module')
            return None, None
        if inputs.shape[0] > 1:
            logger.error(f'Decode mode {self.mode} only supports batch_size=1')

        memory, memory_mask, _, enc_attn_weights = self.encode(inputs, inputs_mask)
        
        batch_size, max_len, hidden_size = memory.size() # (1, max_len, hidden_len)
        
        ctc_probs, memory_lengths = self.model.assistor.inference(memory, memory_mask) # (1, max_len, vocab_size), (1,)
        ctc_probs = ctc_probs.squeeze(0) # (max_len, vocab_size)
        
        cur_hyps = [(tuple(), (0.0, -float('inf')))]
        for t in range(memory_lengths[0]):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(self.beam_width)  # (beam_size,)
            for ps, s in map(lambda x: (x[0].item(), x[1].item()), zip(top_k_logp, top_k_index)):
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = self.log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = self.log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = self.log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = self.log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2 Second beam prune
            next_hyps = sorted(next_hyps.items(),
                               key=lambda x: self.log_add(x[1]), 
                               reverse=True)
            cur_hyps = next_hyps[:self.beam_width]
        max_hpy_len = max(len(hpy[0]) for hpy in cur_hyps)
        
        nbest_probs = torch.ones((1, self.beam_width, max_hpy_len), device=memory.device, dtype=torch.long) * EOS # (B, beam, S)
        nbest_scores = torch.zeros((1, self.beam_width), device=memory.device, dtype=torch.float32) # (B, beam)
        for i in range(self.beam_width):
            for j in range(len(cur_hyps[i][0])):
                nbest_probs[0][i][j] = cur_hyps[i][0][j]
            nbest_scores[0][i] = self.log_add(cur_hyps[i][1])
        
        if not keep_all_result:
            nbest_probs = nbest_probs[:, 0, :].unsqueeze(1)
            nbest_scores = nbest_scores[:, 0].unsqueeze(1)
        
        if require_encoder_out:
            return nbest_probs, nbest_scores, (memory, memory_mask)
        else:
            return nbest_probs, nbest_scores

    def _ctc_rescore(self, inputs, inputs_mask):
        ctc_preds, ctc_scores, encoder_result = self._ctc_beam(inputs, inputs_mask, True, True) # (B, beam, S)
        ctc_preds, ctc_scores = ctc_preds.squeeze(0), ctc_scores.squeeze(0) # (beam, S)

        ctc_preds_in, _ = self.add_bos_eos(ctc_preds, BOS, EOS, PAD)

        memory, memory_mask = encoder_result[0], encoder_result[1]
        memory = memory.repeat([self.beam_width, 1, 1])
        memory_mask = memory_mask.repeat([self.beam_width, 1])
        
        logits, _ = self.model.decoder(ctc_preds_in, memory, memory_mask)
        probs = F.log_softmax(logits, dim=-1)

        best_score = -float('inf')
        best_index = 0
        for i in range(ctc_preds.shape[0]):
            ctc_pred = ctc_preds[i]
            ctc_pred = ctc_pred[ctc_pred != EOS]
            ctc_score = ctc_scores[i]
            decoder_score = 0
            for j in range(ctc_pred.shape[0]):
                w = ctc_pred[j].item()
                decoder_score += probs[i][j][w]
            decoder_score += probs[i][ctc_pred.shape[0]][EOS]
            score = decoder_score + ctc_score * self.ctc_weight
            if score > best_score:
                best_index = i
                best_score = score

        nbest_preds = torch.ones((1, 1, ctc_preds.shape[-1]), device=memory.device, dtype=torch.long)
        nbest_preds[0, 0] = ctc_preds[best_index]

        nbest_scores = torch.tensor([[best_score]], device=memory.device)
        return nbest_preds, nbest_scores


    def recognize(self, inputs, inputs_mask):

        with torch.no_grad():
            # (batch_size, beam_width, max_len), (batch_size, beam_width)
            # TODO: check if ctc module generate sos at the beginning. 
            if self.mode == 'beam':
                nbest_preds, nbest_scores = self._beam(inputs, inputs_mask)
            elif self.mode == 'ctc_greedy':
                nbest_preds, nbest_scores = self._ctc_greedy(inputs, inputs_mask)
            elif self.mode == 'ctc_beam':
                nbest_preds, nbest_scores = self._ctc_beam(inputs, inputs_mask)
            elif self.mode == 'ctc_rescore':
                nbest_preds, nbest_scores = self._ctc_rescore(inputs, inputs_mask)
            else:
                logger.error('Unknown decode mode={}'.format(self.mode))
                nbest_preds, nbest_scores = None, None

        if nbest_preds is None:
            return
        else:
            return self.nbest_translate(nbest_preds), nbest_scores

    def decode_step(self, preds, memory, memory_mask, cache, scores, flag):
        """ decode an utterance in a stepwise way"""
        # score shape of (batch_size * beam_width,)
        batch_size = int(scores.size(0) / self.beam_width)

        # batch_log_probs shape of (batch_size * beam_width, 1, vocab_size)
        batch_log_probs, dec_cache, dec_attn_weights = self.decode(preds, memory, memory_mask, cache['decoder'])

        if self.lm is not None:
            batch_lm_log_probs, lm_hidden = self.lm_decode(preds, cache['lm'])
            batch_lm_log_probs = batch_lm_log_probs.squeeze(1)
            batch_log_probs = batch_log_probs + self.lm_weight * batch_lm_log_probs
            # pdb.set_trace() # FIXME
        else:
            lm_hidden = None

        if batch_log_probs.dim() == 3:
            batch_log_probs = batch_log_probs.squeeze(1)

        last_k_scores, last_k_preds = batch_log_probs.topk(self.beam_width)
        #pdb.set_trace()
        last_k_scores = mask_finished_scores(last_k_scores, flag)
        last_k_preds = mask_finished_preds(last_k_preds, flag)

        # update scores
        scores = scores + last_k_scores
        scores = scores.view(batch_size, self.beam_width * self.beam_width)

        # pruning
        scores, offset_k_indices = torch.topk(scores, k=self.beam_width)
        scores = scores.view(-1, 1)

        device = scores.get_device()
        device = scores.device
        base_k_indices = torch.arange(batch_size, device=device).view(-1, 1).repeat([1, self.beam_width])
        base_k_indices *= self.beam_width ** 2
        best_k_indices = base_k_indices.view(-1) + offset_k_indices.view(-1)

        # update predictions
        best_k_preds = torch.index_select(
            last_k_preds.view(-1), dim=-1, index=best_k_indices)
        if version.parse(torch.__version__) < version.parse('1.6.0'):
            preds_index = best_k_indices.div(self.beam_width)
        else:
            preds_index = best_k_indices.floor_divide(self.beam_width)
        preds_symbol = torch.index_select(
            #preds, dim=0, index=best_k_indices.div(self.beam_width))
            preds, dim=0, index=preds_index)
        preds_symbol = torch.cat(
            (preds_symbol, best_k_preds.view(-1, 1)), dim=1)

        # dec_hidden = reselect_hidden_list(dec_hidden, self.beam_width, best_k_indices)
        # lm_hidden = reselect_hidden_list(lm_hidden, self.beam_width, best_k_indices)

        # finished or not
        end_flag = torch.eq(preds_symbol[:, -1], EOS).view(-1, 1)

        # hidden = {
        #     'decoder': dec_hidden,
        #     'lm': lm_hidden
        # }

        return preds_symbol, cache, scores, end_flag


def mask_finished_scores(score, flag):
    """
    If a sequence is finished, we only allow one alive branch. This function aims to give one branch a zero score
    and the rest -inf score.
    Args:
        score: A real value array with shape [batch_size * beam_size, beam_size].
        flag: A bool array with shape [batch_size * beam_size, 1].
    Returns:
        A real value array with shape [batch_size * beam_size, beam_size].
    """
    beam_width = score.size(-1)
    zero_mask = torch.zeros_like(flag, dtype=torch.bool)
    if beam_width > 1:
        unfinished = torch.cat(
            (zero_mask, flag.repeat([1, beam_width - 1])), dim=1)
        finished = torch.cat(
            (flag.bool(), zero_mask.repeat([1, beam_width - 1])), dim=1)
    else:
        unfinished = zero_mask
        finished = flag.bool()
    score.masked_fill_(unfinished, -float('inf'))
    score.masked_fill_(finished, 0)
    return score


def mask_finished_preds(pred, flag):
    """
    If a sequence is finished, all of its branch should be </S> (3).
    Args:
        pred: A int array with shape [batch_size * beam_size, beam_size].
        flag: A bool array with shape [batch_size * beam_size, 1].
    Returns:
        A int array with shape [batch_size * beam_size].
    """
    beam_width = pred.size(-1)
    finished = flag.repeat([1, beam_width])
    return pred.masked_fill_(finished.bool(), EOS)


def reselect_hidden(tensor, beam_width, indices):
    n_layers, batch_size, hidden_size = tensor.size()
    tensor = tensor.transpose(0, 1).unsqueeze(1).repeat([1, beam_width, 1, 1])
    tensor = tensor.reshape(batch_size * beam_width, n_layers, hidden_size)
    new_tensor = torch.index_select(tensor, dim=0, index=indices)
    new_tensor = new_tensor.transpose(0, 1).contiguous()
    return new_tensor


def reselect_hidden_list(tensor_list, beam_width, indices):

    if tensor_list is None:
        return None

    new_tensor_list = []
    for tensor in tensor_list:
        if isinstance(tensor, tuple):
            h = reselect_hidden(tensor[0], beam_width, indices)
            c = reselect_hidden(tensor[1], beam_width, indices)
            new_tensor_list.append((h, c))
        else:
            new_tensor_list.append(reselect_hidden(tensor, beam_width, indices))
    
    return new_tensor_list
