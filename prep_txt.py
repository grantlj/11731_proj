import os
import sys
import pickle
import pdb
from stanfordcorenlp import StanfordCoreNLP
from collections import Counter
from string import punctuation


nlp = StanfordCoreNLP(r'../stanford-corenlp-full-2018-10-05')
txt_dir = '../TGIF/gt'
output_dir = './gt_parsed'
txt_files = os.listdir(txt_dir)
counter = Counter()
nouns = Counter()
verbs = Counter()
for _, f in enumerate(txt_files):
    if _ % 1000 == 0:
        print(_)
    txt = pickle.load(open(os.path.join(txt_dir, f), 'rb'))
    tags = nlp.pos_tag(txt)
    counter.update([x[0] for x in tags])
    nouns.update([x[0] for x in tags if x[1][:2] == 'NN'])
    verbs.update([x[0] for x in tags if x[1][:2] == 'VB'])
    pickle.dump({'txt': [x[0] for x in tags], 'tags': [x[1] for x in tags], 
                 'nn_id': [x for x in range(len(tags)) if tags[x][1][:2] == 'NN'],
                 'vb_id': [x for x in range(len(tags)) if tags[x][1][:2] == 'VB']},
                open(os.path.join(output_dir, f), 'wb'))

id2word = {}
word2id = {}
id2word[1] = '<start>'
id2word[2] = '<eou>'
word2id['<start>'] = 1
word2id['<eou>'] = 2
ind = 3
for k, v in counter.items():
    id2word[ind] = k
    word2id[k] = ind
    ind += 1

pickle.dump({'id2word': id2word, 'word2id': word2id,
             'nouns': nouns, 'verbs': verbs},
            open('dictionary', 'wb'))
