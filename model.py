import torch
import torch.nn.functional as F
from torch import nn

from config import Config


# TODO: Fix this... 
config = Config.get_debug_config(None)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    def __init__(self, subtoken_input_dim, nodes_vocab_size):
        super(Encoder, self).__init__()

        self.embedding_subtokens = nn.Embedding(subtoken_input_dim, 
                                                config.EMBEDDINGS_SIZE) 
        self.embedding_paths = nn.Embedding(nodes_vocab_size, 
                                            config.EMBEDDINGS_SIZE)

        self.num_layers = 2
        self.lstm = nn.LSTM(config.EMBEDDINGS_SIZE, config.RNN_SIZE//2, 
                            bidirectional=True, 
                            num_layers=self.num_layers,
                            dropout=(1 - config.RNN_DROPOUT_KEEP_PROB),
                            batch_first=True)

        self.lin = nn.Linear(config.EMBEDDINGS_SIZE * 2 + config.RNN_SIZE, 
                             config.DECODER_SIZE, bias=False)

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

        lstm_output, (hidden, cell) = self.lstm(flat_paths) 
        hidden = hidden[-self.num_layers:, :, :]
        hidden = hidden.transpose(0, 1)
        # (batch * max_contexts, rnn_size)
        final_rnn_state = torch.reshape(hidden, (hidden.size()[0], -1))

        # (batch, max_contexts, rnn_size)
        path_aggregated = torch.reshape(final_rnn_state,
                                        (-1, max_context, config.RNN_SIZE))


        # (batch, max_contexts, embed_dim * 2 + rnn_size
        context_embed = torch.cat([start_embed, path_aggregated, end_embed], 
                                  dim=-1)

        # (batch, max_contexts, decoder_size)
        context_embed = torch.tanh(self.lin(context_embed))

        return context_embed


class Decoder(nn.Module):
    def __init__(self, target_input_dim):
        super(Decoder, self).__init__()
        
        self.embedding_target = nn.Embedding(target_input_dim, 
                                             config.DECODER_SIZE)
        self.lstm = nn.LSTMCell(config.DECODER_SIZE, config.DECODER_SIZE)

        self.lin = nn.Linear(config.DECODER_SIZE * 2, 
                             target_input_dim, bias=False)

    def forward(self, seqs, hidden, attention):
        # (batch, max_target, embed_dim) -> (batch, embed_dim)
        emb = self.embedding_target(seqs).squeeze(0)

        _, hidden = self.lstm(emb, hidden)

        # (batch, 2 * decode_size)
        output = torch.cat([hidden, attention], dim=1)
        # (batch, target_input_dim)
        output = torch.tanh(self.lin(output))

        return output, hidden


class Code2Seq(nn.Module):
    def __init__(self, subtoken_input_dim, nodes_input_dim, target_input_dim):
        super(Code2Seq, self).__init__()

        self.target_vocab_size = target_input_dim
        self.encoder = Encoder(subtoken_input_dim, nodes_input_dim)
        self.decoder = Decoder(target_input_dim)

    def attention(self, encode_context, context_mask, hidden):
        h_t, _ = hidden[0]

        # (batch, max_target)
        attn = torch.bmm(encode_context, h_t.unsqueeze(-1)).squeeze(-1)

        # (batch, max_target)
        #n_context_mask = (context_mask == 0).type(torch.float) * -100000
        #attn = attn + n_context_mask
        attn = attn + context_mask

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
        contexts_sum = torch.sum(
            encode_context * context_mask.unsqueeze(-1), dim=1)
                                 
        # (batch, 1)
        context_length = torch.sum(
            context_mask > 0, dim=1, keepdim=True, dtype=torch.float)

        # (batch, decode_size)
        init_state = contexts_sum / context_length

        # (h_t, c_t)
        fake_encoder_state = tuple([init_state, init_state] for _ in
                                   range(config.NUM_DECODER_LAYERS))

        # 1 = BOS
        decoder_input = torch.tensor([1] * config.BATCH_SIZE, 
                                     dtype=torch.long).to(device)
        # (1, batch)
        decoder_input = decoder_input.unsqueeze(0)

        # holds output
        decoder_outputs = torch.zeros(config.MAX_TARGET_PARTS, 
                                      config.BATCH_SIZE, 
                                      self.target_vocab_size).to(device)

        for t in range(config.MAX_TARGET_PARTS):
            attn = self.attention(encode_context, context_mask, fake_encoder_state)
            
            decoder_output, decoder_hidden = self.decoder(decoder_input, 
                                                          *fake_encoder_state, 
                                                          attn)

            decoder_outputs[t] = decoder_output

            if self.training:
                # Does target need to be embedded in some way? :S
                # Something is wrong with the <BOS>-tokens here.. 
                # I add [1] manually assuming that is the index of <BOS>
                padded_tar = torch.tensor([1]+list(target[t])).unsqueeze(0).to(device)
                decoder_input = padded_tar
            else:
                decoder_input = decoder_output.max(-1)[1] 

        return decoder_outputs
