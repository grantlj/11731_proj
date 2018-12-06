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
        self.word_dist = nn.Linear(txt_hidden_size, self.vocab_size)
        self.drop = nn.Dropout(0.5)

        self.hidden_fc = nn.Linear(img_hidden_size, txt_hidden_size)
        self.cell_fc = nn.Linear(img_hidden_size, txt_hidden_size)
        self.loss = nn.NLLLoss(ignore_index=0)

        '''
        for names in self.decoder._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.decoder, name)
                n = bias.size(0)
                start, end = n//4, n//2 
                bias.data[start:end].fill_(10.)
        '''

    def init_hidden(self, src_hidden):
        if type(src_hidden) is tuple:
            hidden = src_hidden[0]
            cell = src_hidden[1]
        else:
            hidden = src_hidden
            cell = src_hidden
        hidden = F.tanh(self.hidden_fc(hidden))
        cell = F.tanh(self.cell_fc(cell))
        return (hidden, cell)

    def forward(self, batch, keep_grad):
        src_sents = batch['video']
        trg_sents = batch['text']
        '''
        pairs = list(zip(src_sents, trg_sents))
        pairs.sort(key=lambda x: x[0].shape[0], reverse=True)
        src_sents, trg_sents = zip(*pairs)
        '''
        #src_lengths = [s.shape[0] for s in src_sents]
        src_lengths = [1 for s in src_sents]
        trg_lengths = [len(s) for s in trg_sents]
        src_max_len = max(src_lengths)
        trg_max_len = max(trg_lengths)
        batch_size = len(src_sents)
        src_ind = torch.zeros(batch_size, self.img_input_size)
        tgt_ind = torch.zeros(batch_size, trg_max_len).long()
        for x in range(len(src_sents)):
            #src_ind[x, :src_sents[x].shape[0]] = torch.from_numpy(src_sents[x])
            src_ind[x] = torch.from_numpy(src_sents[x])
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
                src_encodings, decoder_init_state = self.encode(src_ind, src_lengths)
                loss, num_words = self.decode(src_encodings, src_lengths, decoder_init_state, tgt_ind, trg_lengths)
        return loss, num_words

    def encode(self, src_sents, src_lengths):
        '''
        packed_input = pack_padded_sequence(src_sents, np.asarray(src_lengths), batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden)

        return src_hidden, decoder_hidden
        '''
        decoder_hidden = self.init_hidden(src_sents)
        return None, (decoder_hidden[0].unsqueeze(0), decoder_hidden[1].unsqueeze(0))

    def decode(self, src_encodings, src_lengths, decoder_init_state, tgt_sents, tgt_lengths, keep_grad=True):
        batch_size = decoder_init_state[0].size(1)
        #length = src_encodings.size(1)
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
            decoder_output, decoder_hidden = self.decoder(self.drop(decoder_input.transpose(0, 1)), decoder_hidden)
            #decoder_outputs[:, step, :] = self.word_dist(F.tanh(self.out(torch.cat((decoder_output.transpose(0, 1), context), dim=2)))).squeeze(1)
            #decoder_outputs[:, step, :] = self.word_dist(F.tanh(self.out(decoder_output.transpose(0, 1)))).squeeze(1)
            #decoder_outputs[:, step, :] = self.word_dist(self.out(decoder_output.transpose(0, 1))).squeeze(1)
            #decoder_outputs[:, step, :] = torch.cat((decoder_output.transpose(0, 1), context), dim=2).squeeze(1)
            decoder_outputs[:, step, :] = decoder_output.transpose(0, 1).squeeze(1)
            decoder_input = tgt_embed[:, step+1, :].unsqueeze(1)

        logits = self.word_dist(self.drop(decoder_outputs))
        logits = F.log_softmax(logits, dim=2)
        logits = logits.contiguous().view(-1, self.vocab_size)
        loss = self.loss(logits, tgt_sents[:, 1:].contiguous().view(-1))
        return loss, (tgt_sents[:, 1:] != 0).sum().item()

    def beam_search(self, batch, beam_size, max_decoding_time_step):
        self.eou = 2
        top_k = 5
        src_sents = torch.from_numpy(batch['video'][0]).unsqueeze(0).cuda()
        batch_size = len(src_sents)
        assert batch_size == 1
        src_lengths = np.asarray([src_sents[0].shape[0]])
        '''
        src_embed = torch.from_numpy(src_sents[0]).unsqueeze(0).cuda()
        packed_input = pack_padded_sequence(src_embed, src_lengths, batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden)
        '''
        src_hidden, decoder_hidden = self.encode(src_sents, src_lengths)

        eos_filler = torch.zeros(beam_size).long().cuda().fill_(self.eou)
        decoder_input = self.embed(torch.cuda.LongTensor([1])).unsqueeze(1)
        #length = src_hidden.size(1)
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
        decoder_output = self.word_dist(decoder_output.squeeze(1))
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
            decoder_output = self.word_dist(decoder_output)

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
        self.mos = nn.Linear(txt_hidden_size, 3)
        self.nn_word_dist = nn.Linear(txt_hidden_size, self.vocab_size)
        self.vb_word_dist = nn.Linear(txt_hidden_size, self.vocab_size)

    def forward(self, batch, keep_grad):
        src_sents = batch['video']
        trg_sents = batch['text']
        nn_id = batch['nn_id']
        vb_id = batch['vb_id']
        '''
        pairs = list(zip(src_sents, trg_sents, nn_id))
        pairs.sort(key=lambda x: x[0].shape[0], reverse=True)
        src_sents, trg_sents, nn_id = zip(*pairs)
        '''
        #src_lengths = [s.shape[0] for s in src_sents]
        src_lengths = [1 for s in src_sents]
        trg_lengths = [len(s) for s in trg_sents]
        src_max_len = max(src_lengths)
        trg_max_len = max(trg_lengths)
        batch_size = len(src_sents)
        src_ind = torch.zeros(batch_size, self.img_input_size)
        tgt_ind = torch.zeros(batch_size, trg_max_len).long()
        nn_ids = torch.zeros(batch_size, trg_max_len).byte()
        vb_ids = torch.zeros(batch_size, trg_max_len).byte()
        for x in range(len(src_sents)):
            src_ind[x, :src_sents[x].shape[0]] = torch.from_numpy(src_sents[x])
            tgt_ind[x, :len(trg_sents[x])] = torch.LongTensor(trg_sents[x])
            nn_ids[x, nn_id[x]] = 1
            vb_ids[x, vb_id[x]] = 1
        src_ind = src_ind.cuda()
        tgt_ind = tgt_ind.cuda()
        nn_ids = nn_ids.cuda()
        vb_ids = vb_ids.cuda()

        if keep_grad:
            #   for training stage
            #src_encodings, decoder_init_state = self.encode(src_sents_padded)
            src_encodings, decoder_init_state = self.encode(src_ind, src_lengths)
            loss, num_words, nn_loss, num_nn = self.decode(src_encodings, src_lengths, decoder_init_state, tgt_ind, trg_lengths, nn_ids, vb_ids)
        else:
            #   for test stage
            with torch.no_grad():
                src_encodings, decoder_init_state = self.encode(src_ind, src_lengths, keep_grad=False)
                loss, num_words, nn_loss, num_nn = self.decode(src_encodings, src_lengths, decoder_init_state, tgt_ind, trg_lengths, nn_ids, vb_ids)
        return loss, num_words, nn_loss, num_nn

    def decode(self, src_encodings, src_lengths, decoder_init_state, tgt_sents, tgt_lengths, nn_id, vb_id, keep_grad=True):
        batch_size = decoder_init_state[0].size(1)
        #length = src_encodings.size(1)
        tgt_embed = self.embed(tgt_sents)
        tgt_l = tgt_embed.size(1)
        tags = torch.cuda.LongTensor(batch_size, tgt_l).fill_(0)
        tags[nn_id] = 1
        tags[vb_id] = 2

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
            decoder_output, decoder_hidden = self.decoder(self.drop(decoder_input.transpose(0, 1)), decoder_hidden)
            #decoder_outputs[:, step, :] = self.word_dist(F.tanh(self.out(torch.cat((decoder_output.transpose(0, 1), context), dim=2)))).squeeze(1)
            #decoder_outputs[:, step, :] = self.word_dist(F.tanh(self.out(decoder_output.transpose(0, 1)))).squeeze(1)
            #decoder_outputs[:, step, :] = self.word_dist(self.out(decoder_output.transpose(0, 1))).squeeze(1)
            #decoder_outputs[:, step, :] = torch.cat((decoder_output.transpose(0, 1), context), dim=2).squeeze(1)
            decoder_outputs[:, step, :] = decoder_output.transpose(0, 1).squeeze(1)
            decoder_input = tgt_embed[:, step+1, :].unsqueeze(1)

        logits = self.word_dist(self.drop(decoder_outputs))
        nn_logits = self.nn_word_dist(self.drop(decoder_outputs))
        vb_logits = self.vb_word_dist(self.drop(decoder_outputs))
        mix_prob = F.softmax(self.mos(self.drop(decoder_outputs)), dim=2)
        mix_logits = torch.log(mix_prob)
        probs = F.softmax(logits, dim=2)
        nn_probs = F.softmax(nn_logits, dim=2)
        vb_probs = F.softmax(vb_logits, dim=2)
        total_probs = probs * mix_prob[:, :, [0]] + nn_probs * mix_prob[:, :, [1]] + vb_probs * mix_prob[:, :, [2]]
        logits = torch.log(total_probs + 10e-10)
        logits = logits.contiguous().view(-1, self.vocab_size)
        mix_logits = mix_logits.contiguous().view(-1, 3)
        loss = self.loss(logits, tgt_sents[:, 1:].contiguous().view(-1))
        nn_loss = self.loss(mix_logits, tags[:, 1:].contiguous().view(-1).long())

        return loss, (tgt_sents[:, 1:] != 0).sum().item(), nn_loss, nn_id.sum().item()

    def beam_search(self, batch, beam_size, max_decoding_time_step):
        self.eou = 2
        top_k = 5
        src_sents = torch.from_numpy(batch['video'][0]).unsqueeze(0).cuda()
        batch_size = len(src_sents)
        assert batch_size == 1
        src_lengths = np.asarray([src_sents[0].shape[0]])
        '''
        src_embed = torch.from_numpy(src_sents[0]).unsqueeze(0).cuda()
        packed_input = pack_padded_sequence(src_embed, src_lengths, batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden)
        '''
        src_hidden, decoder_hidden = self.encode(src_sents, src_lengths)

        eos_filler = torch.zeros(beam_size).long().cuda().fill_(self.eou)
        decoder_input = self.embed(torch.cuda.LongTensor([1])).unsqueeze(1)
        #length = src_hidden.size(1)
        src_lengths = torch.cuda.LongTensor(src_lengths)

        #decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2), decoder_hidden)
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        #decoder_output = torch.cat((decoder_output, context), dim=2)
        probs = F.softmax(self.word_dist(decoder_output).squeeze(1), dim=1)
        nn_probs = F.softmax(self.nn_word_dist(decoder_output).squeeze(1), dim=1)
        vb_probs = F.softmax(self.vb_word_dist(decoder_output).squeeze(1), dim=1)
        mix_prob = F.softmax(self.mos(decoder_output).squeeze(1), dim=1)
        total_probs = probs * mix_prob[:, [0]] + nn_probs * mix_prob[:, [1]] + vb_probs * mix_prob[:, [2]]
        total_probs = torch.log(total_probs)

        logprobs, argtop = torch.topk(total_probs, beam_size, dim=1)
        beam = torch.zeros(beam_size, max_decoding_time_step).long().cuda()
        beam[:, 0] = argtop.squeeze(0)
        beam_probs = logprobs.clone().squeeze(0)
        beam_eos = argtop.squeeze(0) == self.eou
        decoder_hidden = (decoder_hidden[0].expand(1, beam_size, self.txt_hidden_size).contiguous(),
                          decoder_hidden[1].expand(1, beam_size, self.txt_hidden_size).contiguous())
        decoder_input = self.embed(argtop.squeeze(0)).unsqueeze(1)

        for t in range(max_decoding_time_step - 1):
            #context = self.attention(decoder_hidden, q_key, q_value, q_mask)
            #decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2).transpose(0, 1), decoder_hidden)
            decoder_output, decoder_hidden = self.decoder(decoder_input.transpose(0, 1), decoder_hidden)
            decoder_output = decoder_output.transpose(0, 1)
            probs = F.softmax(self.word_dist(decoder_output).squeeze(1), dim=1)
            nn_probs = F.softmax(self.nn_word_dist(decoder_output).squeeze(1), dim=1)
            vb_probs = F.softmax(self.vb_word_dist(decoder_output).squeeze(1), dim=1)
            mix_prob = F.softmax(self.mos(decoder_output).squeeze(1), dim=1)
            total_probs = probs * mix_prob[:, [0]] + nn_probs * mix_prob[:, [1]] + vb_probs * mix_prob[:, [2]]
            total_probs = torch.log(total_probs)

            logprobs, argtop = torch.topk(total_probs, top_k, dim=1)
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


class MoS_EXT(Baseline):
    def __init__(self, txt_input_size, img_input_size, txt_hidden_size, img_hidden_size, dictionary, word_vectors=None, context_size=None):
        super(MoS_EXT, self).__init__(txt_input_size, img_input_size, txt_hidden_size, img_hidden_size, dictionary, word_vectors, context_size)
        self.mos = nn.Linear(txt_hidden_size, 3)
        self.nn_out = nn.Linear(2048 + 512 + self.txt_hidden_size, self.txt_input_size)
        self.nn_word_dist = nn.Linear(self.txt_input_size, self.vocab_size)
        self.vb_out = nn.Linear(4096 + self.txt_hidden_size, self.txt_input_size)
        self.vb_word_dist = nn.Linear(self.txt_input_size, self.vocab_size)
        self.nn_mask = torch.cuda.ByteTensor(self.vocab_size).fill_(False)
        self.vb_mask = torch.cuda.ByteTensor(self.vocab_size).fill_(False)
        for word, count in dictionary['nouns'].items():
            self.nn_mask[self.word2id[word]] = 1
        for word, count in dictionary['verbs'].items():
            self.vb_mask[self.word2id[word]] = 1

    def forward(self, batch, keep_grad):
        src_sents = batch['video']
        trg_sents = batch['text']
        nn_id = batch['nn_id']
        vb_id = batch['vb_id']
        motions = batch['mot']
        frames = batch['frame']
        places = batch['place']
        '''
        pairs = list(zip(src_sents, trg_sents, nn_id))
        pairs.sort(key=lambda x: x[0].shape[0], reverse=True)
        src_sents, trg_sents, nn_id = zip(*pairs)
        '''
        src_lengths = [s.shape[0] for s in motions]
        trg_lengths = [len(s) for s in trg_sents]
        src_max_len = max(src_lengths)
        trg_max_len = max(trg_lengths)
        batch_size = len(src_sents)
        src_ind = torch.zeros(batch_size, self.img_input_size)
        mot_feat = torch.zeros(batch_size, 4096)
        frame_feat = torch.zeros(batch_size, 2048)
        place_feat = torch.zeros(batch_size, 512)
        tgt_ind = torch.zeros(batch_size, trg_max_len).long()
        nn_ids = torch.zeros(batch_size, trg_max_len).byte()
        vb_ids = torch.zeros(batch_size, trg_max_len).byte()
        for x in range(len(src_sents)):
            src_ind[x, :src_sents[x].shape[0]] = torch.from_numpy(src_sents[x])
            tgt_ind[x, :len(trg_sents[x])] = torch.LongTensor(trg_sents[x])
            nn_ids[x, nn_id[x]] = 1
            vb_ids[x, vb_id[x]] = 1
            mot_feat[x] = torch.from_numpy(motions[x].sum(axis=0) / src_lengths[x])
            place_feat[x] = torch.from_numpy(places[x].sum(axis=0) / src_lengths[x])
            if len(frames[x].shape) > 1:
                frame_feat[x] = torch.from_numpy(frames[x].sum(axis=0) / src_lengths[x])
            else:
                frame_feat[x] = torch.from_numpy(frames[x])
        src_ind = src_ind.cuda()
        tgt_ind = tgt_ind.cuda()
        nn_ids = nn_ids.cuda()
        vb_ids = vb_ids.cuda()
        mot_feat = mot_feat.cuda()
        frame_feat = frame_feat.cuda()
        place_feat = place_feat.cuda()

        if keep_grad:
            #   for training stage
            #src_encodings, decoder_init_state = self.encode(src_sents_padded)
            src_encodings, decoder_init_state = self.encode(src_ind, src_lengths)
            loss, num_words, nn_loss, num_nn = self.decode(src_encodings, src_lengths, decoder_init_state, tgt_ind, trg_lengths, nn_ids, vb_ids, mot_feat, frame_feat, place_feat)
        else:
            #   for test stage
            with torch.no_grad():
                src_encodings, decoder_init_state = self.encode(src_ind, src_lengths, keep_grad=False)
                loss, num_words, nn_loss, num_nn = self.decode(src_encodings, src_lengths, decoder_init_state, tgt_ind, trg_lengths, nn_ids, vb_ids)
        return loss, num_words, nn_loss, num_nn

    def decode(self, src_encodings, src_lengths, decoder_init_state, tgt_sents, tgt_lengths, nn_id, vb_id, mot_feat, frame_feat, place_feat, keep_grad=True):
        batch_size = decoder_init_state[0].size(1)
        #length = src_encodings.size(1)
        tgt_embed = self.embed(tgt_sents)
        tgt_l = tgt_embed.size(1)
        tags = torch.cuda.LongTensor(batch_size, tgt_l).fill_(0)
        tags[nn_id] = 1
        tags[vb_id] = 2
        frame_feat = frame_feat.expand(tgt_l - 1, batch_size, 2048).permute(1, 0, 2)
        place_feat = place_feat.expand(tgt_l - 1, batch_size, 512).permute(1, 0, 2)
        mot_feat = mot_feat.expand(tgt_l - 1, batch_size, 4096).permute(1, 0, 2)

        decoder_input = tgt_embed[:, 0, :].unsqueeze(1)
        decoder_outputs = torch.cuda.FloatTensor(batch_size, tgt_l - 1, self.txt_hidden_size)
        decoder_hidden = decoder_init_state
        for step in range(tgt_l - 1):
            decoder_output, decoder_hidden = self.decoder(self.drop(decoder_input.transpose(0, 1)), decoder_hidden)
            decoder_outputs[:, step, :] = decoder_output.transpose(0, 1).squeeze(1)
            decoder_input = tgt_embed[:, step+1, :].unsqueeze(1)

        logits = self.word_dist(self.drop(decoder_outputs))
        nn_logits = self.nn_word_dist(F.tanh(self.nn_out(torch.cat((self.drop(decoder_outputs), self.drop(frame_feat), self.drop(place_feat)), dim=2))))
        vb_logits = self.vb_word_dist(F.tanh(self.vb_out(torch.cat((self.drop(decoder_outputs), self.drop(mot_feat)), dim=2))))
        nn_logits[:, :, ~self.nn_mask] = -np.inf
        vb_logits[:, :, ~self.vb_mask] = -np.inf
        mix_prob = F.softmax(self.mos(self.drop(decoder_outputs)), dim=2)
        mix_logits = torch.log(mix_prob)
        probs = F.softmax(logits, dim=2)
        nn_probs = F.softmax(nn_logits, dim=2)
        vb_probs = F.softmax(vb_logits, dim=2)
        total_probs = probs * mix_prob[:, :, [0]] + nn_probs * mix_prob[:, :, [1]] + vb_probs * mix_prob[:, :, [2]]
        logits = torch.log(total_probs + 10e-10)
        logits = logits.contiguous().view(-1, self.vocab_size)
        mix_logits = mix_logits.contiguous().view(-1, 3)
        loss = self.loss(logits, tgt_sents[:, 1:].contiguous().view(-1))
        nn_loss = self.loss(mix_logits, tags[:, 1:].contiguous().view(-1).long())

        return loss, (tgt_sents[:, 1:] != 0).sum().item(), nn_loss, nn_id.sum().item()

    def beam_search(self, batch, beam_size, max_decoding_time_step):
        self.eou = 2
        top_k = 5
        src_sents = torch.from_numpy(batch['video'][0]).unsqueeze(0).cuda()
        motions = torch.from_numpy(batch['mot'][0]).unsqueeze(0).cuda()
        frames = torch.from_numpy(batch['frame'][0]).unsqueeze(0).cuda()
        places = torch.from_numpy(batch['place'][0]).unsqueeze(0).cuda()

        src_lengths = motions[0].shape[0]
        mot_feat = (motions[0].sum(0) / src_lengths).unsqueeze(0)
        place_feat = (places[0].sum(0) / src_lengths).unsqueeze(0)
        if len(frames[0].size()) > 1:
            frame_feat = frames[0].sum(0) / src_lengths
        else:
            frame_feat = frames[0]
        frame_feat = frame_feat.unsqueeze(0)

        batch_size = len(src_sents)
        assert batch_size == 1
        src_lengths = np.asarray([src_sents[0].shape[0]])
        '''
        src_embed = torch.from_numpy(src_sents[0]).unsqueeze(0).cuda()
        packed_input = pack_padded_sequence(src_embed, src_lengths, batch_first=True)
        src_output, src_last_hidden = self.encoder(packed_input)
        src_hidden, _ = pad_packed_sequence(src_output, batch_first=True)
        decoder_hidden = self.init_hidden(src_last_hidden)
        '''
        src_hidden, decoder_hidden = self.encode(src_sents, src_lengths)

        eos_filler = torch.zeros(beam_size).long().cuda().fill_(self.eou)
        decoder_input = self.embed(torch.cuda.LongTensor([1])).unsqueeze(1)
        #length = src_hidden.size(1)
        src_lengths = torch.cuda.LongTensor(src_lengths)

        #decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2), decoder_hidden)
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
        probs = F.softmax(self.word_dist(decoder_output).squeeze(1), dim=1)
        nn_probs = F.softmax(self.nn_word_dist(torch.cat((frame_feat, place_feat), dim=1)).squeeze(1), dim=1)
        vb_probs = F.softmax(self.vb_word_dist(mot_feat).squeeze(1), dim=1)
        mix_prob = F.softmax(self.mos(decoder_output).squeeze(1), dim=1)
        total_probs = probs * mix_prob[:, [0]] + nn_probs * mix_prob[:, [1]] + vb_probs * mix_prob[:, [2]]
        total_probs = torch.log(total_probs)

        logprobs, argtop = torch.topk(total_probs, beam_size, dim=1)
        beam = torch.zeros(beam_size, max_decoding_time_step).long().cuda()
        beam[:, 0] = argtop.squeeze(0)
        beam_probs = logprobs.clone().squeeze(0)
        beam_eos = argtop.squeeze(0) == self.eou
        decoder_hidden = (decoder_hidden[0].expand(1, beam_size, self.txt_hidden_size).contiguous(),
                          decoder_hidden[1].expand(1, beam_size, self.txt_hidden_size).contiguous())
        decoder_input = self.embed(argtop.squeeze(0)).unsqueeze(1)

        for t in range(max_decoding_time_step - 1):
            #context = self.attention(decoder_hidden, q_key, q_value, q_mask)
            #decoder_output, decoder_hidden = self.decoder(torch.cat((decoder_input, context), dim=2).transpose(0, 1), decoder_hidden)
            decoder_output, decoder_hidden = self.decoder(decoder_input.transpose(0, 1), decoder_hidden)
            decoder_output = decoder_output.transpose(0, 1)
            probs = F.softmax(self.word_dist(decoder_output).squeeze(1), dim=1)
            nn_probs = F.softmax(self.nn_word_dist(torch.cat((frame_feat, place_feat), dim=1)).squeeze(1), dim=1)
            vb_probs = F.softmax(self.vb_word_dist(mot_feat).squeeze(1), dim=1)
            mix_prob = F.softmax(self.mos(decoder_output).squeeze(1), dim=1)
            total_probs = probs * mix_prob[:, [0]] + nn_probs * mix_prob[:, [1]] + vb_probs * mix_prob[:, [2]]
            total_probs = torch.log(total_probs)

            logprobs, argtop = torch.topk(total_probs, top_k, dim=1)
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

