import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import pdb


class Baseline(nn.Module):
    def __init__(self, txt_input_size, img_input_size, txt_hidden_size, img_hidden_size, dictionary, word_vectors=None, context_size=None):
        super(Baseline, self).__init__()
        self.txt_input_size = txt_input_size
        self.img_input_size = img_input_size
        self.txt_hidden_size = txt_hidden_size
        self.img_hidden_size = img_hidden_size
        self.id2word = dictionary['id2word']
        self.word2id = dictionary['word2id']
        self.vocab_size = len(self.word2id) + 1
        self.embed = nn.Embedding(self.vocab_size, txt_input_size, padding_idx=0)
        self.encoder = nn.LSTM(img_input_size, img_hidden_size, batch_first=True)
        self.decoder = nn.LSTM(txt_input_size, txt_hidden_size)
        self.out = nn.Linear(txt_hidden_size, txt_input_size)
        self.word_dist = nn.Linear(txt_input_size, self.vocab_size)
        self.word_dist.weight = self.embed.weight

        self.hidden_fc = nn.Linear(img_hidden_size, txt_hidden_size)
        self.cell_fc = nn.Linear(img_hidden_size, txt_hidden_size)
        self.loss = nn.NLLLoss(ignore_index=0)

    def init_hidden(self, src_hidden):
        hidden = src_hidden[0]
        cell = src_hidden[1]
        hidden = F.tanh(self.hidden_fc(hidden))
        cell = F.tanh(self.cell_fc(cell))
        return (hidden, cell)

    def forward(self, batch, keep_grad):
        src_sents = batch['video']
        trg_sents = batch['text']
        pairs = list(zip(src_sents, trg_sents))
        pairs.sort(key=lambda x: len(x[0]), reverse=True)
        src_sents, tgt_sents = zip(*pairs)
        src_lengths = [s.shape[0] for s in src_sents]
        trg_lengths = [len(s) for s in trg_sents]
        src_max_len = max(src_lengths)
        trg_max_len = max(trg_lengths)
        batch_size = len(src_sents)
        src_ind = torch.zeros(batch_size, src_max_len, self.img_input_size)
        tgt_ind = torch.zeros(batch_size, trg_max_len).long()
        for x in range(len(src_sents)):
            src_ind[x, :src_sents[x].shape[0]] = torch.from_numpy(src_sents[x])
            tgt_ind[x, :len(trg_sents[x])] = torch.LongTensor(trg_sents[x])
        src_ind = src_ind.cuda()
        tgt_ind = tgt_ind.cuda()

        if keep_grad:
            #   for training stage
            #src_encodings, decoder_init_state = self.encode(src_sents_padded)
            src_encodings, decoder_init_state = self.encode(src_ind, src_lengths)
            loss, num_words = self.decode(src_encodings, src_lengths, decoder_init_state, tgt_ind, trg_lengths)
        else:
            #   for test stage
            with torch.no_grad():
                src_encodings, decoder_init_state = self.encode(src_ind, src_lengths, keep_grad=False)
                loss, num_words = self.decode(src_encodings, src_lengths, decoder_init_state, tgt_ind, trg_lengths)
        return loss, num_words

    def encode(self, src_sents, src_lengths, keep_grad=True):
        packed_input = pack_padded_sequence(src_sents, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden)

        return src_hidden, decoder_hidden

    def decode(self, src_encodings, src_lengths, decoder_init_state, tgt_sents, tgt_lengths, keep_grad=True):
        batch_size = src_encodings.size(0)
        length = src_encodings.size(1)
        tgt_embed = self.embed(tgt_sents)
        tgt_l = tgt_embed.size(1)

        decoder_input = tgt_embed[:, 0, :].unsqueeze(1)
        decoder_outputs = torch.cuda.FloatTensor(batch_size, tgt_l - 1, self.txt_hidden_size)
        decoder_hidden = decoder_init_state
        '''
        q_key = self.q_key(src_encodings)
        q_value = self.q_value(src_encodings)
        q_mask  = torch.arange(length).long().cuda().repeat(src_encodings.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(length, 1).transpose(0, 1)
        '''
        #src_lengths = torch.cuda.LongTensor(src_lengths)
        for step in range(tgt_l - 1):
            #context = self.attention(decoder_hidden, q_key, q_value, q_mask)
            #decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2).transpose(0, 1), decoder_hidden)
            decoder_output, decoder_hidden = self.decoder(decoder_input.transpose(0, 1), decoder_hidden)
            #decoder_outputs[:, step, :] = self.word_dist(F.tanh(self.out(torch.cat((decoder_output.transpose(0, 1), context), dim=2)))).squeeze(1)
            #decoder_outputs[:, step, :] = self.word_dist(F.tanh(self.out(decoder_output.transpose(0, 1)))).squeeze(1)
            #decoder_outputs[:, step, :] = self.word_dist(self.out(decoder_output.transpose(0, 1))).squeeze(1)
            #decoder_outputs[:, step, :] = torch.cat((decoder_output.transpose(0, 1), context), dim=2).squeeze(1)
            decoder_outputs[:, step, :] = decoder_output.transpose(0, 1).squeeze(1)
            decoder_input = tgt_embed[:, step+1, :].unsqueeze(1)

        logits = self.word_dist(F.tanh(self.out(decoder_outputs)))
        logits = F.log_softmax(logits, dim=2)
        logits = logits.contiguous().view(-1, self.vocab_size)
        loss = self.loss(logits, tgt_sents[:, 1:].contiguous().view(-1))
        return loss, (tgt_sents[:, 1:] != 0).sum().item()

    def beam_search(self, batch, beam_size, max_decoding_time_step):
        self.eou = 2
        top_k = 10
        src_sents = batch['video']
        batch_size = len(src_sents)
        assert batch_size == 1
        src_embed = torch.from_numpy(src_sents[0]).unsqueeze(0).cuda()
        src_lengths = np.asarray([src_sents[0].shape[0]])
        packed_input = pack_padded_sequence(src_embed, src_lengths, batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden)

        eos_filler = torch.zeros(beam_size).long().cuda().fill_(self.eou)
        decoder_input = self.embed(torch.cuda.LongTensor([1])).unsqueeze(1)
        length = src_hidden.size(1)
        src_lengths = torch.cuda.LongTensor(src_lengths)

        '''
        q_key = self.q_key(src_hidden)
        q_value = self.q_value(src_hidden)
        q_mask  = torch.arange(length).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(length, 1).transpose(0, 1)
        context = self.attention(decoder_hidden, q_key, q_value, q_mask)
        '''
        #decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2), decoder_hidden)
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        #decoder_output = torch.cat((decoder_output, context), dim=2)
        decoder_output = self.word_dist(F.tanh(self.out(decoder_output.squeeze(1))))
        decoder_output[:, 0] = -np.inf

        logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), beam_size, dim=1)
        beam = torch.zeros(beam_size, max_decoding_time_step).long().cuda()
        beam[:, 0] = argtop
        beam_probs = logprobs.clone().squeeze(0)
        beam_eos = argtop.squeeze(0) == self.eou
        decoder_hidden = (decoder_hidden[0].expand(1, beam_size, self.txt_hidden_size).contiguous(),
                          decoder_hidden[1].expand(1, beam_size, self.txt_hidden_size).contiguous())
        decoder_input = self.embed(argtop.squeeze(0)).unsqueeze(1)
        '''
        src_hidden = src_hidden.expand(beam_size, length, self.hidden_size * (int(self.bi_direct) + 1))
        q_key = self.q_key(src_hidden)
        q_value = self.q_value(src_hidden)
        q_mask  = torch.arange(length).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(length, 1).transpose(0, 1)
        '''

        for t in range(max_decoding_time_step - 1):
            #context = self.attention(decoder_hidden, q_key, q_value, q_mask)
            #decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2).transpose(0, 1), decoder_hidden)
            decoder_output, decoder_hidden = self.decoder(decoder_input.transpose(0, 1), decoder_hidden)
            decoder_output = decoder_output.transpose(0, 1)
            decoder_output = self.word_dist(F.tanh(self.out(decoder_output)))

            logprobs, argtop = torch.topk(F.log_softmax(decoder_output.squeeze(1), dim=1), top_k, dim=1)
            best_probs, best_args = (beam_probs.expand(top_k, beam_size).transpose(0, 1) + logprobs).view(-1).topk(beam_size)

            last = best_args / top_k
            curr = best_args % top_k
            beam[:, :] = beam[last, :]
            beam_eos = beam_eos[last]
            beam_probs = beam_probs[last]
            beam[:, t+1] = argtop[last, curr] * (~beam_eos).long() + eos_filler * beam_eos.long()
            mask = ~beam_eos
            beam_probs[mask] = (beam_probs[mask] * (t+1) + best_probs[mask]) / (t+2)
            decoder_hidden = (decoder_hidden[0][:, last, :], decoder_hidden[1][:, last, :])

            beam_eos = beam_eos | (beam[:, t+1] == self.eou)
            decoder_input = self.embed(beam[:, t+1]).unsqueeze(1)

            if beam_eos.all():
                break

        best, best_arg = beam_probs.max(0)
        translation = beam[best_arg].cpu().tolist()
        if self.eou in translation:
            translation = translation[:translation.index(self.eou)]
        translation = [self.id2word[w] for w in translation]
        return translation


class MoS(Baseline):
    def __init__(self, txt_input_size, img_input_size, txt_hidden_size, img_hidden_size, dictionary, word_vectors=None, context_size=None):
        super(MoS, self).__init__(txt_input_size, img_input_size, txt_hidden_size, img_hidden_size, dictionary, word_vectors, context_size)
        self.mos = nn.Linear(txt_hidden_size, 2)
        self.nn_word_dist = nn.Linear(txt_input_size, self.vocab_size)

    def forward(self, batch, keep_grad):
        src_sents = batch['video']
        trg_sents = batch['text']
        nn_tags = []
        pdb.set_trace()
        pairs = list(zip(src_sents, trg_sents))
        pairs.sort(key=lambda x: len(x[0]), reverse=True)
        src_sents, tgt_sents = zip(*pairs)
        src_lengths = [s.shape[0] for s in src_sents]
        trg_lengths = [len(s) for s in trg_sents]
        src_max_len = max(src_lengths)
        trg_max_len = max(trg_lengths)
        batch_size = len(src_sents)
        src_ind = torch.zeros(batch_size, src_max_len, self.img_input_size)
        tgt_ind = torch.zeros(batch_size, trg_max_len).long()
        for x in range(len(src_sents)):
            src_ind[x, :src_sents[x].shape[0]] = torch.from_numpy(src_sents[x])
            tgt_ind[x, :len(trg_sents[x])] = torch.LongTensor(trg_sents[x])
        src_ind = src_ind.cuda()
        tgt_ind = tgt_ind.cuda()

        if keep_grad:
            #   for training stage
            #src_encodings, decoder_init_state = self.encode(src_sents_padded)
            src_encodings, decoder_init_state = self.encode(src_ind, src_lengths)
            loss, num_words = self.decode(src_encodings, src_lengths, decoder_init_state, tgt_ind, trg_lengths)
        else:
            #   for test stage
            with torch.no_grad():
                src_encodings, decoder_init_state = self.encode(src_ind, src_lengths, keep_grad=False)
                loss, num_words = self.decode(src_encodings, src_lengths, decoder_init_state, tgt_ind, trg_lengths)
        return loss, num_words

    def decode(self, src_encodings, src_lengths, decoder_init_state, tgt_sents, tgt_lengths, keep_grad=True):
        batch_size = src_encodings.size(0)
        length = src_encodings.size(1)
        tgt_embed = self.embed(tgt_sents)
        tgt_l = tgt_embed.size(1)

        decoder_input = tgt_embed[:, 0, :].unsqueeze(1)
        decoder_outputs = torch.cuda.FloatTensor(batch_size, tgt_l - 1, self.txt_hidden_size)
        decoder_hidden = decoder_init_state
        '''
        q_key = self.q_key(src_encodings)
        q_value = self.q_value(src_encodings)
        q_mask  = torch.arange(length).long().cuda().repeat(src_encodings.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(length, 1).transpose(0, 1)
        '''
        #src_lengths = torch.cuda.LongTensor(src_lengths)
        for step in range(tgt_l - 1):
            #context = self.attention(decoder_hidden, q_key, q_value, q_mask)
            #decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2).transpose(0, 1), decoder_hidden)
            decoder_output, decoder_hidden = self.decoder(decoder_input.transpose(0, 1), decoder_hidden)
            #decoder_outputs[:, step, :] = self.word_dist(F.tanh(self.out(torch.cat((decoder_output.transpose(0, 1), context), dim=2)))).squeeze(1)
            #decoder_outputs[:, step, :] = self.word_dist(F.tanh(self.out(decoder_output.transpose(0, 1)))).squeeze(1)
            #decoder_outputs[:, step, :] = self.word_dist(self.out(decoder_output.transpose(0, 1))).squeeze(1)
            #decoder_outputs[:, step, :] = torch.cat((decoder_output.transpose(0, 1), context), dim=2).squeeze(1)
            decoder_outputs[:, step, :] = decoder_output.transpose(0, 1).squeeze(1)
            decoder_input = tgt_embed[:, step+1, :].unsqueeze(1)

        logits = self.word_dist(F.tanh(self.out(decoder_outputs)))
        logits = F.log_softmax(logits, dim=2)
        logits = logits.contiguous().view(-1, self.vocab_size)
        loss = self.loss(logits, tgt_sents[:, 1:].contiguous().view(-1))
        return loss, (tgt_sents[:, 1:] != 0).sum().item()

    def beam_search(self, batch, beam_size, max_decoding_time_step):
        self.eou = 2
        top_k = 10
        src_sents = batch['video']
        batch_size = len(src_sents)
        assert batch_size == 1
        src_embed = torch.from_numpy(src_sents[0]).unsqueeze(0).cuda()
        src_lengths = np.asarray([src_sents[0].shape[0]])
        packed_input = pack_padded_sequence(src_embed, src_lengths, batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden)

        eos_filler = torch.zeros(beam_size).long().cuda().fill_(self.eou)
        decoder_input = self.embed(torch.cuda.LongTensor([1])).unsqueeze(1)
        length = src_hidden.size(1)
        src_lengths = torch.cuda.LongTensor(src_lengths)

        '''
        q_key = self.q_key(src_hidden)
        q_value = self.q_value(src_hidden)
        q_mask  = torch.arange(length).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(length, 1).transpose(0, 1)
        context = self.attention(decoder_hidden, q_key, q_value, q_mask)
        '''
        #decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2), decoder_hidden)
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        #decoder_output = torch.cat((decoder_output, context), dim=2)
        decoder_output = self.word_dist(F.tanh(self.out(decoder_output.squeeze(1))))
        decoder_output[:, 0] = -np.inf

        logprobs, argtop = torch.topk(F.log_softmax(decoder_output, dim=1), beam_size, dim=1)
        beam = torch.zeros(beam_size, max_decoding_time_step).long().cuda()
        beam[:, 0] = argtop
        beam_probs = logprobs.clone().squeeze(0)
        beam_eos = argtop.squeeze(0) == self.eou
        decoder_hidden = (decoder_hidden[0].expand(1, beam_size, self.txt_hidden_size).contiguous(),
                          decoder_hidden[1].expand(1, beam_size, self.txt_hidden_size).contiguous())
        decoder_input = self.embed(argtop.squeeze(0)).unsqueeze(1)
        '''
        src_hidden = src_hidden.expand(beam_size, length, self.hidden_size * (int(self.bi_direct) + 1))
        q_key = self.q_key(src_hidden)
        q_value = self.q_value(src_hidden)
        q_mask  = torch.arange(length).long().cuda().repeat(src_hidden.size(0), 1) < torch.cuda.LongTensor(src_lengths).repeat(length, 1).transpose(0, 1)
        '''

        for t in range(max_decoding_time_step - 1):
            #context = self.attention(decoder_hidden, q_key, q_value, q_mask)
            #decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2).transpose(0, 1), decoder_hidden)
            decoder_output, decoder_hidden = self.decoder(decoder_input.transpose(0, 1), decoder_hidden)
            decoder_output = decoder_output.transpose(0, 1)
            decoder_output = self.word_dist(F.tanh(self.out(decoder_output)))

            logprobs, argtop = torch.topk(F.log_softmax(decoder_output.squeeze(1), dim=1), top_k, dim=1)
            best_probs, best_args = (beam_probs.expand(top_k, beam_size).transpose(0, 1) + logprobs).view(-1).topk(beam_size)

            last = best_args / top_k
            curr = best_args % top_k
            beam[:, :] = beam[last, :]
            beam_eos = beam_eos[last]
            beam_probs = beam_probs[last]
            beam[:, t+1] = argtop[last, curr] * (~beam_eos).long() + eos_filler * beam_eos.long()
            mask = ~beam_eos
            beam_probs[mask] = (beam_probs[mask] * (t+1) + best_probs[mask]) / (t+2)
            decoder_hidden = (decoder_hidden[0][:, last, :], decoder_hidden[1][:, last, :])

            beam_eos = beam_eos | (beam[:, t+1] == self.eou)
            decoder_input = self.embed(beam[:, t+1]).unsqueeze(1)

            if beam_eos.all():
                break

        best, best_arg = beam_probs.max(0)
        translation = beam[best_arg].cpu().tolist()
        if self.eou in translation:
            translation = translation[:translation.index(self.eou)]
        translation = [self.id2word[w] for w in translation]
        return translation

