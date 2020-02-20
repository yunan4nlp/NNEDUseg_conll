import torch.nn.functional as F
from modules.Layer import *

class EDUSegmenter(object):
    def __init__(self, pwordEnc, wordLSTM, dec, crf, config):
        self.config = config
        self.pwordEnc = pwordEnc
        self.wordLSTM = wordLSTM
        self.dec = dec
        self.crf = crf
        self.use_cuda = next(filter(lambda p: p.requires_grad, wordLSTM.parameters())).is_cuda

    def train(self):
        self.pwordEnc.train()
        self.wordLSTM.train()
        self.dec.train()
        self.crf.train()
        self.training = True

    def eval(self):
        self.pwordEnc.eval()
        self.wordLSTM.eval()
        self.dec.eval()
        self.crf.eval()
        self.training = False

    def encode(self, bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask, word_mask):
        if self.use_cuda:
            bert_indice_input = bert_indice_input.cuda()
            bert_segments_ids = bert_segments_ids.cuda()
            bert_piece_ids = bert_piece_ids.cuda()
            bert_mask = bert_mask.cuda()
            word_mask = word_mask.cuda()
            #sent_extwords = sent_extwords.cuda()
        #x_extword_embed = self.pwordEnc(edu_extwords)
        sent_extword_embed = self.pwordEnc(bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask)
        self.encoder_output = self.wordLSTM(sent_extword_embed, word_mask)

    def decode(self, label_mask):
        if self.use_cuda:
            label_mask = label_mask.cuda()
        self.decoder_output = self.dec(self.encoder_output)

        output = self.crf.viterbi_tags(self.decoder_output, label_mask)
        best_paths = []
        for path, score in output:
            best_paths.append(path)
        return best_paths

    def compute_accuracy(self, predict_seg, gold_seg):
        total_num = 0
        correct = 0
        batch_size = len(predict_seg)
        assert batch_size == len(gold_seg)
        for b_iter in range(batch_size):
            sent_len = len(predict_seg[b_iter])
            assert sent_len == len(gold_seg[b_iter])
            for cur_step in range(sent_len):
                if predict_seg[b_iter][cur_step] == gold_seg[b_iter][cur_step]:
                    correct += 1
                total_num += 1
        return total_num, correct

    def compute_loss(self, true_segs, label_mask):
        if self.use_cuda:
            label_mask = label_mask.cuda()
        batch_size, sent_len, label_size = self.decoder_output.size()
        true_segs = _model_var(
            self.wordLSTM,
            pad_sequence(true_segs, length=sent_len, padding=0, dtype=np.int64))
        crf_loss = -self.crf(self.decoder_output, true_segs, label_mask) / batch_size
        return crf_loss

def _model_var(model, x):
    p = next(filter(lambda p: p.requires_grad, model.parameters()))
    if p.is_cuda:
        x = x.cuda(p.get_device())
    return torch.autograd.Variable(x)

def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)

