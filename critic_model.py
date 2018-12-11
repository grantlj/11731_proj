import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Baseline(nn.Module):
    def __init__(self, txt_input_size, img_input_size, txt_hidden_size, img_hidden_size, dictionary,fast_cider):
        super(Baseline, self).__init__()
        self.txt_input_size = txt_input_size
        self.img_input_size = img_input_size
        self.txt_hidden_size = txt_hidden_size
        self.img_hidden_size = img_hidden_size
        self.id2word = dictionary['id2word']
        self.word2id = dictionary['word2id']
        self.vocab_size = len(self.word2id) + 1

        #   all the params
        self.embed = nn.Embedding(self.vocab_size, txt_input_size, padding_idx=0)
        self.encoder = nn.LSTM(img_input_size, img_hidden_size, batch_first=True)
        self.decoder = nn.LSTM(txt_input_size, txt_hidden_size)
        self.word_dist = nn.Linear(txt_hidden_size, self.vocab_size)
        self.drop = nn.Dropout(0.5)

        self.faster_cider=fast_cider

        self.hidden_fc = nn.Linear(img_hidden_size, txt_hidden_size)
        self.cell_fc = nn.Linear(img_hidden_size, txt_hidden_size)
        self.loss=nn.NLLLoss()

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


    #   teacher forcing decoding
    def __decode_by_common__(self, decoder_init_state, tgt_sents):
        batch_size = decoder_init_state[0].size(1)
        #length = src_encodings.size(1)
        tgt_embed = self.embed(tgt_sents)
        tgt_l = tgt_embed.size(1)

        decoder_input = tgt_embed[:, 0, :].unsqueeze(1)
        decoder_outputs = torch.cuda.FloatTensor(batch_size, tgt_l - 1, self.txt_hidden_size)
        decoder_hidden = decoder_init_state

        for step in range(tgt_l - 1):
            decoder_output, decoder_hidden = self.decoder(self.drop(decoder_input.transpose(0, 1)), decoder_hidden)
            decoder_outputs[:, step, :] = decoder_output.transpose(0, 1).squeeze(1)

            #   teacher forcing mode
            decoder_input = tgt_embed[:, step+1, :].unsqueeze(1)

        logits = self.word_dist(self.drop(decoder_outputs))
        logits = F.log_softmax(logits, dim=2)
        logits = logits.contiguous().view(-1, self.vocab_size)
        loss = self.loss(logits, tgt_sents[:, 1:].contiguous().view(-1))
        return loss, (tgt_sents[:, 1:] != 0).sum().item()

    #   sampling during the decoding step
    def __decoder_by_sample__(self,decoder_init_state,tgt_sents):
        batch_size = decoder_init_state[0].size(1)
        tgt_embed = self.embed(tgt_sents)
        tgt_l = tgt_embed.size(1)
        decoder_input = tgt_embed[:, 0, :].unsqueeze(1)

        decoder_hidden = decoder_init_state

        sampled_sents=[]
        sample_inputs=[]

        for step in range(tgt_l-1):
            sample_inputs.append(decoder_input.detach().cpu().numpy())

            #   decoder_input: batch_size*1*300
            decoder_output, decoder_hidden = self.decoder(self.drop(decoder_input.transpose(0, 1)), decoder_hidden)

            #   decoder_output: 1*batch_size*512, decoder_logits: 1*batch_size*12204
            decoder_logits = self.word_dist(self.drop(decoder_output))
            decoder_logits = F.softmax(decoder_logits, dim=2)

            #   decoder_logits: 16*12204 [current step's prob for each batch]
            decoder_probs = decoder_logits.contiguous().view(-1, self.vocab_size)
            decoder_sampler=torch.distributions.categorical.Categorical(probs=decoder_probs)

            #   decoder_sample_ind: [16], tokens sampled at current step
            decoder_sample_ind=decoder_sampler.sample()
            sampled_sents.append(decoder_sample_ind.detach().cpu().numpy())

            #   selecting the next inputs embedding according to sampled instance (should be batch*1*300)
            decoder_input=self.embed(decoder_sample_ind).unsqueeze(dim=1)

            #print "[DEBUG], step:",step,"..."

        #   sample_sents: sent_len*batch_size; sample_inputs: sent_len*batch-size*1*300  (sempled sentences, and embeddings)
        sampled_sents=np.asarray(sampled_sents);sample_sent_embed=np.squeeze(np.asarray(sample_inputs))
        return sampled_sents,sample_sent_embed

    #   greedy based decoding
    def __decoder_by_greedy__(self,decoder_init_state,tgt_sents):
        batch_size = decoder_init_state[0].size(1)
        tgt_embed = self.embed(tgt_sents)
        tgt_l = tgt_embed.size(1)
        decoder_input = tgt_embed[:, 0, :].unsqueeze(1)

        decoder_hidden = decoder_init_state

        greedy_sents=[]
        greedy_inputs=[]

        for step in range(tgt_l-1):
            greedy_inputs.append(decoder_input.detach().cpu().numpy())

            #   decoder_input: batch_size*1*300
            decoder_output, decoder_hidden = self.decoder(self.drop(decoder_input.transpose(0, 1)), decoder_hidden)

            #   decoder_output: 1*batch_size*512, decoder_logits: 1*batch_size*12204
            decoder_logits = self.word_dist(self.drop(decoder_output))
            decoder_logits = F.softmax(decoder_logits, dim=2)

            #   decoder_logits: 16*12204 [current step's prob for each batch]
            decoder_probs = decoder_logits.contiguous().view(-1, self.vocab_size)
            decoder_greedy_ind=torch.max(decoder_probs,dim=1)[1]

            greedy_sents.append(decoder_greedy_ind.detach().cpu().numpy())

            #   selecting the next inputs embedding according to sampled instance (should be batch*1*300)
            decoder_input = self.embed(decoder_greedy_ind).unsqueeze(dim=1)


            #rint "[DEBUG], step:", step,"..."

        greedy_sents = np.asarray(greedy_sents);greedy_sent_embed = np.squeeze(np.asarray(greedy_inputs))
        return greedy_sents,greedy_sent_embed


    def __multi_mode_forward__(self,batch,mode="sample",sample_gt=None):

        assert mode in ["sample","greedy","common","sample_as_gt"]

        src_sents = batch['video']

        if mode=="sample_as_gt" and not sample_gt is None:
            trg_sents=sample_gt
        else:
            trg_sents = batch['text']

        # src_lengths = [s.shape[0] for s in src_sents]
        src_lengths = [1 for s in src_sents]
        trg_lengths = [len(s) for s in trg_sents]
        src_max_len = max(src_lengths)
        trg_max_len = max(trg_lengths)
        batch_size = len(src_sents)
        src_ind = torch.zeros(batch_size, self.img_input_size)
        tgt_ind = torch.zeros(batch_size, trg_max_len).long()

        for x in range(len(src_sents)):
            src_ind[x] = torch.from_numpy(src_sents[x])
            tgt_ind[x, :len(trg_sents[x])] = torch.LongTensor(trg_sents[x])

        src_ind = src_ind.cuda()
        tgt_ind = tgt_ind.cuda()

        #   finished the encoding step
        src_encodings, decoder_init_state = self.encode(src_ind, src_lengths)

        #   now we running the decoding step according to different modes
        if mode=="sample":
            sample_sents,sample_sent_embed=self.__decoder_by_sample__(decoder_init_state,tgt_sents=tgt_ind)
            return sample_sents,sample_sent_embed

        elif mode=="greedy":
            greedy_sents,greedy_sent_embed=self.__decoder_by_greedy__(decoder_init_state,tgt_sents=tgt_ind)
            return greedy_sents,greedy_sent_embed

        elif mode=="common":
            common_loss,num_words=self.__decode_by_common__(decoder_init_state,tgt_sents=tgt_ind)
            return common_loss,num_words
        elif mode=="sample_as_gt":
            common_loss,num_words=self.__decode_by_common__(decoder_init_state,tgt_sents=tgt_ind)
            return common_loss,num_words
        else:
            raise NotImplementedError

    def __covert_dn_to_nd__(self,org_list):
        ret_list=[]
        sent_len=len(org_list)
        batch_len=len(org_list[0])
        for bid in xrange(0,batch_len):
            cur_s=[]
            for sid in xrange(0,sent_len):
                cur_s.append(org_list[sid][bid])
            ret_list.append(cur_s)
        return ret_list

    def __convert_ind_sent_to_text__(self,ind_sents):
        ret_texts=[]
        for cur_sent in ind_sents:
            text_tokens=[]
            for tid in cur_sent:
                try:
                    text_tokens.append(self.id2word[tid])
                except:
                    continue
                if tid == 2:
                    break
            ret_texts.append(" ".join(text_tokens))
        return ret_texts

    def __calc_cider__(self,raw_pred_sents,id_list,raw_gt_sents):
        #   raw_gt_sents: in n*di format
        #   raw pred sents: in max(d)*n format
        raw_pred_sents=self.__covert_dn_to_nd__(raw_pred_sents)

        #   add <s> </s> if needed
        for i in xrange(0,len(raw_pred_sents)):
            if  raw_pred_sents[i][0]!=1:
                raw_pred_sents[i].insert(0,1)
            if raw_pred_sents[i][1]!=2:
                raw_pred_sents[i].append(2)

        raw_pred_sents=self.__convert_ind_sent_to_text__(raw_pred_sents)
        raw_gt_sents=self.__convert_ind_sent_to_text__(raw_gt_sents)

        #   filter out the <start> and <eou>
        for i in xrange(len(raw_pred_sents)):
            raw_pred_sents[i]=str(raw_pred_sents[i])
            raw_gt_sents[i]=str(raw_gt_sents[i])
            raw_pred_sents[i]=raw_pred_sents[i].replace("<start> ","")
            raw_pred_sents[i]=raw_pred_sents[i].replace(" <eou>","")
            raw_gt_sents[i]=raw_gt_sents[i].replace("<start> ", "")
            raw_gt_sents[i]=raw_gt_sents[i].replace(" <eou>", "")

        score, scores = self.faster_cider.compute_cider(raw_pred_sents, id_list)
        #score=0;scores=0
        return score,scores,raw_pred_sents

    def __append_start_end__(self,tid_list):
        ret_list=[]
        for cur_tids in tid_list:
            if not cur_tids[0]==1:
                cur_tids=[1]+cur_tids
            if not 2 in cur_tids:
                cur_tids.append(2)
            first2=cur_tids.index(2)
            cur_tids=cur_tids[0:first2+1]
            ret_list.append(cur_tids)
        return ret_list

    def forward(self, batch, keep_grad):

        cur_ids=batch['id_list']
        cur_gts=batch['text']

        sample_sents,sample_sent_embed=self.__multi_mode_forward__(batch,mode="sample")
        greedy_sents,greedy_sent_embed=self.__multi_mode_forward__(batch, mode="greedy")

        #   use sample as the ground-truth to calculate another loss
        sample_as_gt_loss, num_words = self.__multi_mode_forward__(batch, mode="sample_as_gt", sample_gt=self.__append_start_end__((self.__covert_dn_to_nd__(sample_sents))))

        sample_batch_cider,sample_ciders,sample_readable_sents=self.__calc_cider__(sample_sents,cur_ids,cur_gts)
        greedy_batch_cider,greedy_ciders,greedy_readable_sents=self.__calc_cider__(greedy_sents,cur_ids,cur_gts)

        #   calculate the reinforce A
        A=sample_batch_cider-greedy_batch_cider;

        #   jia's choice (need a negative sign here)
        A=-A.item()

        loss=A*sample_as_gt_loss
        return loss, num_words,A,greedy_readable_sents,-greedy_batch_cider

    def encode(self, src_sents, src_lengths):
        decoder_hidden = self.init_hidden(src_sents)
        return None, (decoder_hidden[0].unsqueeze(0), decoder_hidden[1].unsqueeze(0))

    def decode(self, src_encodings, src_lengths, decoder_init_state, tgt_sents, tgt_lengths, keep_grad=True):
        #return self.__decode_by_common__(decoder_init_state,tgt_sents=tgt_sents)
        pass

    def beam_search(self, batch, beam_size, max_decoding_time_step):
        self.eou = 2
        top_k = 5
        src_sents = torch.from_numpy(batch['video'][0]).unsqueeze(0).cuda()
        batch_size = len(src_sents)
        assert batch_size == 1
        src_lengths = np.asarray([src_sents[0].shape[0]])

        src_hidden, decoder_hidden = self.encode(src_sents, src_lengths)

        eos_filler = torch.zeros(beam_size).long().cuda().fill_(self.eou)
        decoder_input = self.embed(torch.cuda.LongTensor([1])).unsqueeze(1)

        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

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

        for t in range(max_decoding_time_step - 1):

            decoder_output, decoder_hidden = self.decoder(decoder_input.transpose(0, 1), decoder_hidden)
            decoder_output = decoder_output.transpose(0, 1)
            decoder_output = self.word_dist(decoder_output)

            logprobs, argtop = torch.topk(F.log_softmax(decoder_output.squeeze(1), dim=1), top_k, dim=1)
            best_probs, best_args = (beam_probs.expand(top_k, beam_size).transpose(0, 1) + logprobs).view(-1).topk(
                beam_size)

            last = best_args / top_k
            curr = best_args % top_k
            beam[:, :] = beam[last, :]
            beam_eos = beam_eos[last]
            beam_probs = beam_probs[last]
            beam[:, t + 1] = argtop[last, curr] * (~beam_eos).long() + eos_filler * beam_eos.long()
            mask = ~beam_eos
            beam_probs[mask] = (beam_probs[mask] * (t + 1) + best_probs[mask]) / (t + 2)
            decoder_hidden = (decoder_hidden[0][:, last, :], decoder_hidden[1][:, last, :])

            beam_eos = beam_eos | (beam[:, t + 1] == self.eou)
            decoder_input = self.embed(beam[:, t + 1]).unsqueeze(1)

            if beam_eos.all():
                break

        best, best_arg = beam_probs.max(0)
        translation = beam[best_arg].cpu().tolist()
        if self.eou in translation:
            translation = translation[:translation.index(self.eou)]
        translation = [self.id2word[w] for w in translation]
        return translation

