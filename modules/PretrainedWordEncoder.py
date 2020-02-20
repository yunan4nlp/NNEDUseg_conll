from modules.Layer import *
from modules.ScaleMix import *
import torch.nn as nn

class PretrainedWordEncoder(nn.Module):
    def __init__(self, config, model, input_dims, layer_num):
        super(PretrainedWordEncoder, self).__init__()
        self.config = config
        self.word_dims = config.word_dims
        self.layer_num = layer_num
        self.input_dims = input_dims
        self.pretrain_model = model

        self.mlp_words = nn.ModuleList([NonLinear(self.input_dims, self.word_dims, activation=GELU()) \
                                        for i in range(self.layer_num)])

        #self.mlp_words = nn.ModuleList([nn.Linear(in_features=self.input_dims,
                                                  #out_features=self.word_dims, bias=True) \
                                        #for i in range(self.layer_num)])
        self.rescale = ScalarMix(mixture_size=self.layer_num)

    def forward(self, bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask=None):
        outputs = self.pretrain_model(bert_indice_input, bert_segments_ids,
                                          bert_piece_ids, bert_mask)
        proj_hiddens = []
        for idx, input in enumerate(outputs):
            cur_hidden = self.mlp_words[idx](input)
            proj_hiddens.append(cur_hidden)
        x_embed = self.rescale(proj_hiddens)
        return x_embed
