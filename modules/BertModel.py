from transformers.modeling_bert import *


class MyBertModel(BertModel):
    def __init__(self, config):
        super(MyBertModel, self).__init__(config)

    def forward(self, bert_indices, bert_segments, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(bert_indices)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(bert_indices, token_type_ids=bert_segments)
        last_output, encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)
        return encoder_outputs


class BertExtractor(nn.Module):
    def __init__(self, config):
        super(BertExtractor, self).__init__()
        self.config = config
        self.bert = MyBertModel.from_pretrained(config.bert_dir)
        self.bert.encoder.output_hidden_states = config.output_hidden_states
        self.bert.encoder.output_attentions = config.output_attentions
        self.bert_hidden_size = self.bert.config.hidden_size
        self.bert_layers = self.bert.config.num_hidden_layers + 1
        self.start_layer = config.start_layer
        self.end_layer = config.end_layer
        if self.start_layer > self.bert_layers - 1: self.start_layer = self.bert_layers - 1
        self.layer_num = self.end_layer - self.start_layer

        for p in self.bert.named_parameters():
            # print(p[0])
            p[1].requires_grad = False

    def forward(self, bert_indices, bert_segments, bert_pieces, bert_mask):
        outputs = []
        with torch.no_grad():
            all_outputs = self.bert(bert_indices, bert_segments, bert_mask)
            for idx in range(self.start_layer, self.end_layer):
                cur_output = torch.bmm(bert_pieces, all_outputs[idx])
                outputs.append(cur_output)

        return outputs

