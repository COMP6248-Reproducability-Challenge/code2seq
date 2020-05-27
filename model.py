import torch
import torch.nn.functional as F
from torch import nn

from config import Config
from common import Common


# TODO: Fix this... 
config = Config.get_default_config(None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, subtoken_input_dim, nodes_vocab_size):
        super(Encoder, self).__init__()

        self.embedding_subtokens = nn.Embedding(subtoken_input_dim, 
                                                config.EMBEDDINGS_SIZE) 
        self.embedding_paths = nn.Embedding(nodes_vocab_size, 
                                            config.EMBEDDINGS_SIZE)

        self.num_layers = 1
        self.lstm = nn.LSTM(config.EMBEDDINGS_SIZE, config.RNN_SIZE//2, 
                            bidirectional=True, 
                            num_layers=self.num_layers,
                            dropout=(1 - config.RNN_DROPOUT_KEEP_PROB),
                            batch_first=True)

        self.lin = nn.Linear(config.EMBEDDINGS_SIZE * 2 + config.RNN_SIZE, 
                             config.DECODER_SIZE, bias=False)
        self.norm = nn.LayerNorm(config.DECODER_SIZE)

    def forward(self, start_leaf, ast_path, end_leaf, start_leaf_mask, 
                end_leaf_mask, ast_path_lengths):
        # (batch, max_context, max_name_parts, embed_dim)
        start_embed = self.embedding_subtokens(start_leaf)
        end_embed = self.embedding_subtokens(end_leaf)

        # (batch, max_contexts, max_path_length+1, embed_dim)
        path_embed = self.embedding_paths(ast_path)

        # (batch, max_contexts, max_name_parts, 1)
        end_leaf_mask = end_leaf_mask.unsqueeze(-1)
        start_leaf_mask = start_leaf_mask.unsqueeze(-1)

        # (batch, max_contexts, embed_dim)
        start_embed = torch.sum(start_embed * start_leaf_mask, dim=2)
        end_embed = torch.sum(end_embed * end_leaf_mask, dim=2)

        max_context = path_embed.size()[1]
        # (batch * max_contexts, max_path_lenght+1, embed_dim)
        flat_paths = path_embed.view(-1, config.MAX_PATH_LENGTH, 
                                      config.EMBEDDINGS_SIZE)

        # TODO: Clean this up
        ast_path_lengths[ast_path_lengths == 0] = 1

        ordered_len, ordered_idx = ast_path_lengths.view(
            config.BATCH_SIZE  * config.MAX_CONTEXTS, ).sort(0, descending=True)
        path_embed_packed = nn.utils.rnn.pack_padded_sequence(
            flat_paths, ordered_len, batch_first=True)

        lstm_output, (hidden, cell) = self.lstm(path_embed_packed)

        # (batch * max_contexts, rnn_size)
        hidden = hidden.view(config.BATCH_SIZE * config.MAX_CONTEXTS, config.RNN_SIZE)

        final_rnn_state = torch.index_select(hidden, 0, ordered_idx)
        # (batch, max_context, rnn_size)
        final_rnn_state = final_rnn_state.view(config.BATCH_SIZE,
                                               config.MAX_CONTEXTS,
                                               config.RNN_SIZE)


        # (batch, max_contexts, embed_dim * 2 + rnn_size
        context_embed = torch.cat([start_embed, final_rnn_state, end_embed],
                                  dim=-1)

        # (batch, max_contexts, decoder_size)
        context_embed = self.lin(context_embed)
        context_embed = self.norm(context_embed)
        context_embed = torch.tanh(context_embed)

        return context_embed


class Decoder(nn.Module):
    def __init__(self, target_input_dim):
        super(Decoder, self).__init__()
        

        self.num_layers = 2
        self.lstm = nn.LSTMCell(config.EMBEDDINGS_SIZE, config.DECODER_SIZE)

        self.input_lin = nn.Linear(config.DECODER_SIZE * 2,
                                   config.DECODER_SIZE)
        self.norm = nn.LayerNorm(config.DECODER_SIZE)
        self.output_lin = nn.Linear(config.DECODER_SIZE,
                                    target_input_dim,
                                    bias=False)


    def forward(self, input_, hidden, encode_out, context_mask, attention):
        # (batch, max_target, embed_dim) -> (batch, embed_dim)
        #emb = self.embedding_target(seqs).squeeze(0)

        hidden, cell = self.lstm(input_, hidden)

        # (batch, decode_size)
        attention = attention(encode_out, context_mask, hidden)

        # (batch, 2 * decode_size)
        output = torch.cat([attention, hidden], dim=1)
        # (batch, target_input_dim)
        output = self.input_lin(output)
        output = self.norm(output)
        output = torch.tanh(output)
        output = self.output_lin(output).unsqueeze(1)

        return output, (hidden, cell)


class Code2Seq(nn.Module):
    def __init__(self, dictionaries):
        super(Code2Seq, self).__init__()

        self.dict_ = dictionaries
        self.embedding_target = nn.Embedding(self.dict_.target_vocab_size,
                                             config.EMBEDDINGS_SIZE)
        self.encoder = Encoder(self.dict_.subtoken_vocab_size,
                               self.dict_.nodes_vocab_size)
        self.decoder = Decoder(self.dict_.target_vocab_size)

    def attention(self, encode_context, context_mask, decode_out):
        # (batch, max_target)
        attn = torch.bmm(encode_context, decode_out.unsqueeze(-1)).squeeze(-1)

        # -10000 to remove activation from non-attended areas
        # (batch, max_target)
        n_context_mask = (context_mask == 0).type(torch.float) * -100000
        attn = attn + n_context_mask

        attn_weight = F.softmax(attn, dim=1)
        # (batch, decode_size)
        attn = torch.bmm(attn_weight.unsqueeze(1), encode_context).squeeze(1)
        
        return attn 
        
    def forward(self, start_leaf, ast_path, end_leaf, target, start_leaf_mask, 
                end_leaf_mask, target_mask, context_mask, ast_path_lengths):
        encode_context = self.encoder(start_leaf, ast_path, end_leaf, 
                                      start_leaf_mask, end_leaf_mask, 
                                      ast_path_lengths)

        # (batch, decode_size)
        contexts_sum = torch.sum(encode_context, dim=1)
                                 
        # (batch, 1)
        context_length = torch.sum(
            context_mask > 0, dim=1, keepdim=True, dtype=torch.float)

        # (batch, decode_size)
        init_state = contexts_sum / context_length

        h_t = init_state.clone()
        c_t = init_state.clone()
        decoder_hidden = (h_t, c_t)

        # (h_t, c_t)
        #decoder_hidden = tuple([init_state, init_state] for _ in
                               #range(config.NUM_DECODER_LAYERS))

        # Empty input to decoder, only containing start-of-sequence tag
        SOS_token = self.dict_.target_to_index[Common.SOS]
        decoder_input = torch.tensor([SOS_token] * config.BATCH_SIZE, 
                                     dtype=torch.long).to(device)
        # (1, batch)
        decoder_input = decoder_input.unsqueeze(0)

        # holds output
        decoder_outputs = torch.zeros(config.BATCH_SIZE,
                                      1,
                                      self.dict_.target_vocab_size).to(device)
        target = self.embedding_target(target)

        decoder_input = target[:, 0]
        for t in range(config.MAX_TARGET_PARTS):
            decoder_output, decoder_hidden = self.decoder(decoder_input, 
                                                          decoder_hidden, 
                                                          encode_context,
                                                          context_mask,
                                                          self.attention)
            decoder_outputs = torch.cat([decoder_outputs, decoder_output],
                                        dim=1)

            if self.training:
                decoder_input = target[:, t+1]
            else:
                output = torch.softmax(decoder_output, dim=1)
                output = torch.argmax(output, dim=1)
                decoder_input = self.embedding_target(output)

        return decoder_outputs

    def get_evaluation(self, predicted, targets):
        true_positive, false_positive, false_negative = 0, 0, 0

        predicted = torch.argmax(predicted, dim=2)
        for pred, targ in zip(predicted, targets):
            for word in pred:
                if Common.word_not_meta_token(word, self.dict_.target_to_index):
                    if word in targ: true_positive += 1
                    else: false_positive += 1
            for word in targ:
                if Common.word_not_meta_token(word, self.dict_.target_to_index):
                    if word not in pred: false_negative += 1

        return true_positive, false_positive, false_negative
