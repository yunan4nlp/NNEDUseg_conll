from modules.Layer import *

class Decoder(nn.Module):
    def __init__(self, vocab, config):
        super(Decoder, self).__init__()
        self.config = config

        self.mlp = NonLinear(input_size=config.lstm_hiddens * 2,
                             hidden_size=config.hidden_size,
                             activation=nn.Tanh())

        self.output = nn.Linear(in_features=config.hidden_size,
                                out_features=vocab.seglabel_size,
                                bias=False)

        torch.nn.init.kaiming_uniform_(self.output.weight)

    def forward(self, hidden_state):
        mlp_hidden = self.mlp(hidden_state)
        if self.training:
            mlp_hidden = drop_sequence_sharedmask(mlp_hidden, self.config.dropout_mlp)
        labels_score = self.output(mlp_hidden)
        return labels_score
