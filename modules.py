# -*- coding: utf-8 -*-
"""
 @Time    : 2020/1/23 下午10:19
 @FileName: modules.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

layer_norm = nn.LayerNorm


class MultiHeadBlockSelf(nn.Module):
    def __init__(self, n_input, n_head=12):
        super().__init__()
        self.combined_projection = nn.Linear(n_input, 2 * (n_input // n_head) * n_head + (n_input // 2) * n_head)
        self.output_projection = nn.Linear((n_input // 2) * n_head, n_input)
        self.mlp = nn.Sequential(nn.Linear(n_input, n_input),
                                 nn.SELU(inplace=True),
                                 nn.Linear(n_input, n_input),
                                 )
        nn.init.xavier_normal_(self.combined_projection.weight, gain=0.1)
        nn.init.xavier_normal_(self.output_projection.weight, gain=0.1)
        # nn.init.xavier_normal_(self.inter.weight, gain=0.1)

        self._scale = (n_input // n_head) ** 0.5
        self.att_dim = (n_input // n_head) * n_head
        self.num_heads = n_head

        # self.dropout = nn.Dropout(p=0.1)
        self.ln = layer_norm(n_input)
        # self.ln = nn.LayerNorm(n_input)

    def forward(self, representations, mask):
        batch_size, timesteps, _ = representations.size()
        combined_projection = F.leaky_relu(self.combined_projection(representations), inplace=True)
        queries, keys, *values = combined_projection.split(self.att_dim, -1)
        queries = queries.contiguous()
        keys = keys.contiguous()
        values = torch.cat(values, -1).contiguous()

        values_per_head = values.view(batch_size, timesteps, self.num_heads, values.size(-1) // self.num_heads)
        values_per_head = values_per_head.transpose(1, 2).contiguous()
        values_per_head = values_per_head.view(batch_size * self.num_heads, timesteps,
                                               values.size(-1) // self.num_heads)

        queries_per_head = queries.view(batch_size, timesteps, self.num_heads, self.att_dim // self.num_heads)
        queries_per_head = queries_per_head.transpose(1, 2).contiguous()
        queries_per_head = queries_per_head.view(batch_size * self.num_heads, timesteps,
                                                 self.att_dim // self.num_heads)

        keys_per_head = keys.view(batch_size, timesteps, self.num_heads, self.att_dim // self.num_heads)
        keys_per_head = keys_per_head.transpose(1, 2).contiguous()
        keys_per_head = keys_per_head.view(batch_size * self.num_heads, timesteps, self.att_dim // self.num_heads)

        similarities = queries_per_head.bmm(keys_per_head.transpose(2, 1)) / self._scale

        similarities = F.softmax(similarities.masked_fill(mask, -np.inf), 2)

        outputs = similarities.bmm(values_per_head)

        outputs = outputs.view(batch_size, self.num_heads, timesteps, values.size(-1) // self.num_heads)
        # shape (batch_size, timesteps, num_heads, values_dim/num_heads)
        outputs = outputs.transpose(1, 2).contiguous()
        # shape (batch_size, timesteps, values_dim)
        outputs = outputs.view(batch_size, timesteps, values.size(-1))

        inter = self.ln(representations + F.leaky_relu(self.output_projection(outputs), inplace=True))

        hidden = F.gelu(self.mlp(inter))

        return inter + hidden

    def inference(self, previous_representations, current_representations):
        batch_size, timesteps, _ = current_representations.size()
        combined_projection = F.leaky_relu(self.combined_projection(current_representations), inplace=True)
        queries, keys, *values = combined_projection.split(self.att_dim, -1)
        queries = queries.contiguous()
        keys = keys.contiguous()
        values = torch.cat(values, -1).contiguous()

        values_per_head = values.view(batch_size, timesteps, self.num_heads, values.size(-1) // self.num_heads)
        values_per_head = values_per_head.transpose(1, 2).contiguous()
        values_per_head = values_per_head.view(batch_size * self.num_heads, timesteps,
                                               values.size(-1) // self.num_heads)

        queries_per_head = queries.view(batch_size, timesteps, self.num_heads, self.att_dim // self.num_heads)
        queries_per_head = queries_per_head.transpose(1, 2).contiguous()
        queries_per_head = queries_per_head.view(batch_size * self.num_heads, timesteps,
                                                 self.att_dim // self.num_heads)

        keys_per_head = keys.view(batch_size, timesteps, self.num_heads, self.att_dim // self.num_heads)
        keys_per_head = keys_per_head.transpose(1, 2).contiguous()
        keys_per_head = keys_per_head.view(batch_size * self.num_heads, timesteps, self.att_dim // self.num_heads)

        [p_k, p_v] = previous_representations

        p_k = torch.cat([p_k, keys_per_head], 1) if p_k is not None else keys_per_head
        p_v = torch.cat([p_v, values_per_head], 1) if p_v is not None else values_per_head

        similarities = queries_per_head.bmm(p_k.transpose(2, 1)) / self._scale

        similarities = F.softmax(similarities, 2)

        outputs = similarities.bmm(p_v)

        outputs = outputs.view(batch_size, self.num_heads, timesteps, values.size(-1) // self.num_heads)
        # shape (batch_size, timesteps, num_heads, values_dim/num_heads)
        outputs = outputs.transpose(1, 2).contiguous()
        # shape (batch_size, timesteps, values_dim)
        outputs = outputs.view(batch_size, timesteps, values.size(-1))

        inter = self.ln(current_representations + F.leaky_relu(self.output_projection(outputs), inplace=True))

        hidden = F.gelu(self.mlp(inter).float()).type_as(inter)

        return p_k, p_v, inter + hidden

    def get_initiate_representations(self, representations, mask):
        batch_size, timesteps, _ = representations.size()
        combined_projection = F.leaky_relu(self.combined_projection(representations), inplace=True)
        queries, keys, *values = combined_projection.split(self.att_dim, -1)
        queries = queries.contiguous()
        keys = keys.contiguous()
        values = torch.cat(values, -1).contiguous()

        values_per_head = values.view(batch_size, timesteps, self.num_heads, values.size(-1) // self.num_heads)
        values_per_head = values_per_head.transpose(1, 2).contiguous()
        values_per_head = values_per_head.view(batch_size * self.num_heads, timesteps,
                                               values.size(-1) // self.num_heads)

        queries_per_head = queries.view(batch_size, timesteps, self.num_heads, self.att_dim // self.num_heads)
        queries_per_head = queries_per_head.transpose(1, 2).contiguous()
        queries_per_head = queries_per_head.view(batch_size * self.num_heads, timesteps,
                                                 self.att_dim // self.num_heads)

        keys_per_head = keys.view(batch_size, timesteps, self.num_heads, self.att_dim // self.num_heads)
        keys_per_head = keys_per_head.transpose(1, 2).contiguous()
        keys_per_head = keys_per_head.view(batch_size * self.num_heads, timesteps, self.att_dim // self.num_heads)

        similarities = queries_per_head.bmm(keys_per_head.transpose(2, 1)) / self._scale

        similarities = F.softmax(similarities.masked_fill(mask, -np.inf), 2)

        outputs = similarities.bmm(values_per_head)

        outputs = outputs.view(batch_size, self.num_heads, timesteps, values.size(-1) // self.num_heads)
        # shape (batch_size, timesteps, num_heads, values_dim/num_heads)
        outputs = outputs.transpose(1, 2).contiguous()
        # shape (batch_size, timesteps, values_dim)
        outputs = outputs.view(batch_size, timesteps, values.size(-1))

        inter = self.ln(representations + F.leaky_relu(self.output_projection(outputs), inplace=True))

        hidden = F.gelu(self.mlp(inter))

        output = inter + hidden
        return keys_per_head, values_per_head, output


class Attention(nn.Module):
    def __init__(self, n_hidden, n_layer, n_head=6):
        super().__init__()
        self.n_head = n_head
        self.self_att = nn.ModuleList()
        self.ln1 = nn.ModuleList()
        for _ in range(n_layer):
            self_att = MultiHeadBlockSelf(n_hidden, n_head)
            ln1 = layer_norm(n_hidden)
            self.self_att.append(self_att)
            self.ln1.append(ln1)
        self.output_ln = layer_norm(n_hidden)

    def forward(self, representations, mask):
        for self_att, ln1 in zip(self.self_att, self.ln1):
            representations = ln1(representations)
            representations = self_att(representations, mask)
        return self.output_ln(representations)

    def inference(self, previous_representations, current_representations):
        num = 0
        for self_att, ln1 in zip(self.self_att, self.ln1):
            current_representations = ln1(current_representations)
            [p_k, p_v] = previous_representations[num]
            p_k, p_v, current_representations = self_att.inference([p_k, p_v], current_representations)
            previous_representations[num] = [p_k, p_v]
            num += 1
        return self.output_ln(current_representations)

    def get_initiate_representations(self, representations, mask):
        previous_representations = []
        for self_att, ln1 in zip(self.self_att, self.ln1):
            representations = ln1(representations)
            p_k, p_v, representations = self_att.get_initiate_representations(representations, mask)
            previous_representations.append([p_k, p_v])
        return previous_representations, self.output_ln(representations)


class AttentionShare(Attention):
    def __init__(self, n_hidden, n_layer, n_head):
        super().__init__(n_hidden, n_layer, n_head)
        self.self_att = nn.ModuleList()
        self.ln1 = nn.ModuleList()
        self_att = MultiHeadBlockSelf(n_hidden, n_head)
        for _ in range(n_layer):
            ln1 = layer_norm(n_hidden)
            self.self_att.append(self_att)
            self.ln1.append(ln1)


class Project(nn.Module):
    def __init__(self, n_hidden, n_embedding, vocabulary_size):
        super().__init__()
        self.project = nn.Linear(n_hidden, n_embedding)
        self.output = nn.Linear(n_embedding, vocabulary_size, bias=False)

    def forward(self, hidden):
        return self.output(F.leaky_relu(self.project(hidden), inplace=True))


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embedding, n_hidden, n_layer, n_head=12):
        super().__init__()
        self.n_head = n_head
        vocabulary_size = (2 + vocab_size // 8) * 8
        self.vocabulary_size = vocabulary_size
        self.word_embedding = nn.Embedding(vocabulary_size, embedding_dim=n_embedding)
        self.n_embedding = n_embedding
        self.encoder = nn.RNN(input_size=n_embedding, hidden_size=n_hidden, batch_first=True, bidirectional=False)
        self.attention = AttentionShare(n_hidden, n_layer, n_head=n_head)
        self.project = Project(n_hidden, n_embedding, vocabulary_size)
        self.project.output.weight = self.word_embedding.weight

    def get_attention_representations(self, source):
        # if self._pos_ids is None:

        word_embedding = self.word_embedding(source)
        encoder_representations, _ = self.encoder(word_embedding)

        len_s = encoder_representations.size(1)
        b_size = encoder_representations.size(0)
        subsequent_mask = torch.triu(
            torch.ones((len_s, len_s), device=source.device, dtype=torch.bool), diagonal=1)
        mask = subsequent_mask.unsqueeze(0).expand(b_size * self.n_head, -1, -1)  # b x ls x ls
        encoder_representations = self.attention(encoder_representations, mask)
        return encoder_representations


    @staticmethod
    def top_k_logits(logits, k):
        if k == 0:
            return logits
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1]
        return torch.where(logits < min_values.unsqueeze(1), torch.ones_like(logits, dtype=logits.dtype) * -1e10,
                           logits)

    def forward(self, source, size=32):
        word_embedding = self.word_embedding(source)
        encoder_representations, rnn_hidden = self.encoder(word_embedding)
        len_s = encoder_representations.size(1)
        b_size = encoder_representations.size(0)
        subsequent_mask = torch.triu(
            torch.ones((len_s, len_s), device=source.device, dtype=torch.bool), diagonal=1)
        # subsequent_mask = subsequent_mask.float().masked_fill(subsequent_mask == 1, float('-inf')).masked_fill(
        #     subsequent_mask == 0, float(0.0))
        mask = subsequent_mask.unsqueeze(0).expand(b_size * self.n_head, -1, -1)  # b x ls x ls
        previous_representations, encoder_representations = self.attention.get_initiate_representations(
            encoder_representations, mask)
        encoder_representations = encoder_representations[:, -1:, :]
        output = []
        for i in range(size):
            logits = self.project(encoder_representations) / 0.9
            top_k = self.top_k_logits(logits.squeeze(1), 100)
            probs = F.softmax(top_k, 1)
            word_ids = torch.multinomial(probs, 1)
            # word_ids = torch.argmax(logits, -1)
            output.append(word_ids)
            word_embedding = self.word_embedding(word_ids)
            current_representations, rnn_hidden = self.encoder(word_embedding, rnn_hidden)
            encoder_representations = self.attention.inference(previous_representations, current_representations)
        return torch.cat(output, 1)
