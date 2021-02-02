import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, encoder_dim, attention_dim, hidden_size, dropout):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.attention_dim = attention_dim
        self.vocab_size = vocab_size

        self.attention = nn.Linear(encoder_dim + embedding_dim, attention_dim)

        self.rnn = nn.GRU(attention_dim, hidden_size)

        self.out = nn.Linear(hidden_size + attention_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden, encoder_outputs):
        concated_inputs = torch.cat((inputs, encoder_outputs.unsqueeze(1)), dim=2)
        attented = self.dropout(torch.tanh(self.attention(concated_inputs))).permute(1, 0, 2)

        output, hidden = self.rnn(attented, hidden)
        concated_outputs = F.relu(torch.cat((output, attented), dim=2))
        out = self.out(self.dropout(concated_outputs))

        return out, hidden

    def initHidden(self, bs):
        return torch.zeros(1, bs, self.hidden_size, device="cuda")


class Img2Caption(nn.Module):
    def __init__(self, encoder, decoder, gptModel):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.gpt = gptModel

    def forward(self, image, caption, teacher_forcing_ratio=0.5):
        #         caption: bs х max_len
        batch_size = caption.shape[0]
        max_len = caption.shape[1]
        trg_vocab_size = self.decoder.vocab_size
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size)
        encoder_outputs = self.encoder(image)
        caption = self.gpt(caption).last_hidden_state

        first_input = caption[:, 0].unsqueeze(1)
        hidden = self.decoder.initHidden(first_input.shape[0])

        for t in range(1, max_len):
            output, hidden = self.decoder(first_input, hidden, encoder_outputs)
            outputs[t] = output

            teacher_force = np.random.random() < teacher_forcing_ratio
            if teacher_force:
                first_input = caption[:, t].unsqueeze(1)
            else:
                first_input = self.gpt(torch.argmax(output, dim=2)).last_hidden_state.permute(1, 0, 2)

        return outputs