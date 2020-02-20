from modules.Layer import *

class WordLSTM(nn.Module):
    def __init__(self, vocab, config):
        super(WordLSTM, self).__init__()
        self.config = config
        #self.conv = nn.Conv2d(1, config.hidden_size, (config.cnn_window, config.word_dims),
                              #padding=(config.cnn_window//2, 0), bias=True)

        self.lstm = MyLSTM(
            input_size=config.word_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in = config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

    def forward(self, x_extword_embed, masks):
        if self.training:
            x_extword_embed = drop_sequence_sharedmask(x_extword_embed, self.config.dropout_emb)

        #x_extword_embed = x_extword_embed.unsqueeze(1)
        #hidden = torch.tanh(self.conv(x_extword_embed))
        #hidden = hidden.squeeze(-1).transpose(1, 2)

        outputs, _ = self.lstm(x_extword_embed, masks, None)
        outputs = outputs.transpose(1, 0)
        #if self.training:
            #outputs = drop_sequence_sharedmask(outputs, self.config.dropout_mlp)
        return outputs # batch, EDU_num, EDU_len, hidden
