import torch
import math
from otrans.data import EOS, PAD, BLK
from torch.nn.utils.rnn import pad_sequence

class Recognizer():
    def __init__(self, model, idx2unit=None, lm=None, lm_weight=None, ngpu=1):

        self.ngpu = ngpu

        self.model = model
        self.model.eval()
        if self.ngpu > 0 : self.model.cuda()

        self.lm = lm
        if self.lm is not None:
            self.lm.eval()
            if self.ngpu > 0: self.lm.eval()

        self.idx2unit = idx2unit

        self.lm_weight = lm_weight

    def recognize(self, inputs, inputs_length):
        raise NotImplementedError

    def lm_decode(self, preds, hidden=None):
        """
        Args:
            preds: [batch_size, lens]
            hidde: [time_step, batch_size, hidden_size] or ([time_step, batch_size, hidden_size], [time_step, batch_size, hidden_size])
        """
        if self.lm.model_type == 'transformer_lm':
            log_probs = self.lm.predict(preds, last_frame=True)
        elif self.lm.model_type == 'bertpretrain_lm':
            log_probs = self.lm.predict(preds)
        else:
            preds = preds[:, -1].unsqueeze(-1)
            log_probs, hidden = self.lm.predict(preds, hidden)
        return log_probs, hidden

    def lm_decode_with_index(self, preds, index, hidden=None):
        """
        Args:
            preds: [batch_size, lens]
            hidde: [time_step, batch_size, hidden_size] or ([time_step, batch_size, hidden_size], [time_step, batch_size, hidden_size])
        """
        if self.lm.model_type == 'transformer_lm':
            log_probs = self.lm.predict(preds, last_frame=False)
            log_probs = select_tensor_based_index(log_probs, index)
        else:
            preds = select_tensor_based_index(preds, index).unsqueeze(-1)
            log_probs, hidden = self.lm.predict(preds, hidden)
        return log_probs, hidden

    def lm_rescoring(self, preds, pred_lens):
        # preds [beam_size, lens]
        # preds_len [beam_size]

        if self.lm.model_type == 'transformer_lm':
            log_probs = self.lm.predict(preds, last_frame=False)
        else:
            log_probs = []
            hidden = None
            for t in range(preds.size(1)):
                log_prob, hidden = self.lm.predict(preds[:, t].unsqueeze(-1), hidden)
                log_probs.append(log_prob)

            log_probs = torch.cat(log_probs, dim=1)

        rescores = []
        max_length = log_probs.size(1)
        vocab_size = log_probs.size(-1)

        for b in range(preds.size(0)):
            base_index = torch.arange(max_length, device=preds.device)
            bias_index = preds[b].reshape(-1)

            index = base_index * vocab_size + bias_index
            score = torch.index_select(log_probs[b].reshape(-1), dim=-1, index=index)

            label_len = min(int(pred_lens[b]), score.size(0))
            score[label_len-1:] = 0
            rescores.append(torch.sum(score) / label_len)

        rescores = torch.FloatTensor(rescores)
        _, indices = torch.sort(rescores, dim=-1, descending=True)

        sorted_preds = preds[indices] 
        sorted_length = pred_lens[indices]

        return sorted_preds, sorted_length          

    def translate(self, seqs):
        results = []
        for seq in seqs:
            pred = []
            for i in seq:
                if int(i) == EOS:
                    break
                if int(i) == PAD:
                    continue
                pred.append(self.idx2unit[int(i)])
            results.append(' '.join(pred))
        return results

    def nbest_translate(self, nbest_preds):
        """translate index to charactor
        Args:
            nbest_preds (torch.Tensor): (batch_size, nbest, lens), EOS marks an end of a sentence.
        
        Returns:
            results List[List[List[int]]]: (batch_size, nbest)
        """
        assert nbest_preds.dim() == 3
        batch_size, nbest, lens = nbest_preds.size()
        results = []
        for b in range(batch_size):
            nbest_list = []
            for n in range(nbest):
                pred = []
                for i in range(lens):
                    token = int(nbest_preds[b, n, i])
                    if token == EOS:
                        break
                    pred.append(self.idx2unit[token])
                nbest_list.append(' '.join(pred))
            results.append(nbest_list)
        return results

    def make_pad_mask(self, lengths, max_len):
        """Make mask tensor containing indices of padded part.
        See description of make_non_pad_mask.
        Args:
            lengths (torch.Tensor): Batch of lengths (B,).
        Returns:
            torch.Tensor: Mask tensor containing indices of padded part.
        Examples:
            >>> lengths = [5, 3, 2]
            >>> make_pad_mask(lengths)
            masks = [[0, 0, 0, 0 ,0],
                    [0, 0, 0, 1, 1],
                    [0, 0, 1, 1, 1]]
        """
        batch_size = lengths.size(0)
        max_len = max_len if max_len > 0 else lengths.max().item()
        seq_range = torch.arange(0,
                                max_len,
                                dtype=torch.int64,
                                device=lengths.device)
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_length_expand = lengths.unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand
        return mask

    def remove_dup_and_blank(self, pred):
        ctc_pred = torch.ones_like(pred) * EOS
        ori_ptr = 0
        ctc_ptr = 0
        while ori_ptr < pred.shape[0]:
            if pred[ori_ptr] != BLK:
                ctc_pred[ctc_ptr] = pred[ori_ptr]
                ctc_ptr += 1
            prev = pred[ori_ptr]
            while ori_ptr < pred.shape[0] and pred[ori_ptr] == prev:
                ori_ptr += 1
        return ctc_pred

    def add_bos_eos(self, labels, bos, eos, pad):
        _sos = torch.tensor([bos], dtype=labels.dtype, device=labels.device)
        _eos = torch.tensor([eos], dtype=labels.dtype, device=labels.device)
        ret = pad_sequence([
            torch.cat((_sos, label[(label != pad) & (label != bos) & (label != eos)], _eos)) for label in labels
        ], True, pad)
        return ret[:, :-1], ret[:, 1:]

    def log_add(self, probs):
        """
        Stable log add : mainly handle the -inf overflow problem
        """
        if all(prob == -float('inf') for prob in probs):
            return -float('inf')
        
        max_prob = max(probs)
        t_probs_add = math.log(sum(math.exp(prob - max_prob) for prob in probs))
        return max_prob + t_probs_add


def select_tensor_based_index(tensor, index):
    # tensor: [b, c, t, v]
    # index: [b]
    # return [b, t, v]
    assert tensor.dim() >= 2
    assert index.dim() == 1

    batch_size = tensor.size(0)
    tensor_len = tensor.size(1)

    base_index = torch.arange(batch_size, device=tensor.device) * tensor_len
    indices = base_index + index

    if tensor.dim() == 2:
        select_tensor = torch.index_select(tensor.reshape(batch_size * tensor_len), 0, indices.long())
    else:
        assert tensor.dim() == 3
        select_tensor = torch.index_select(tensor.reshape(batch_size * tensor_len, tensor.size(-1)), 0, indices.long())

    return select_tensor
