# -*- coding: utf-8 -*-
"""
 @Time    : 2020/4/2 上午11:45
 @FileName: interface.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import argparse

import sentencepiece as spm
import torch

from modules import GPT
from utils import clean, padding

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default='input.txt')
parser.add_argument("--output", type=str, default='output.txt')
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--input_max_length", type=int, default=32)
parser.add_argument("--output_max_length", type=int, default=256)
args = parser.parse_args()
sp = spm.SentencePieceProcessor()
sp.load('full.unigram.32768.model')
n_embedding = 256
n_hidden = 2048
n_layer = 12
n_head = 16
batch_size = 8
vocab_size = sp.GetPieceSize()
model = GPT(vocab_size, n_embedding, n_hidden, n_layer, n_head)
print('model build done!')

with open('model/model.gpt.{}.{}.th'.format(n_hidden, n_layer),
          'rb') as f:
    model.load_state_dict(torch.load(f, map_location='cpu'))
model.eval()
if args.cuda:
    model.cuda()


def process_one(one):
    ids = sp.encode_as_ids(clean(one))
    return ids[0:args.input_max_length]


def clean_prediction(prediction):
    if vocab_size + 1 in prediction:
        end = prediction.index(vocab_size + 1)
    elif 0 in prediction:
        end = prediction.index(0)
    else:
        end = len(prediction)
    prediction = prediction[0:end]
    return sp.decode_ids([x for x in prediction if x < vocab_size])


data = []
with open(args.input, encoding='utf-8') as f:
    for line in f:
        data.append([len(data), process_one(line.strip())])

print('data size {}'.format(len(data)))

data = sorted(data, key=lambda x: len(x[1]))
predictions = []
with torch.no_grad():
    total = len(data)
    for i in range(0, total, batch_size):
        input_ids = [[1]+x[1] for x in data[i:i + batch_size]]
        input_index = [x[0] for x in data[i:i + batch_size]]
        input_ids, _ = padding(input_ids, max_len=args.input_max_length)
        input_ids = torch.LongTensor(input_ids)
        if args.cuda:
            input_ids = input_ids.cuda()
        output = model(input_ids, args.output_max_length).cpu().numpy().tolist()
        output = [clean_prediction(x) for x in output]
        predictions.extend(list(zip(input_index, output)))

print('generate done!!')

predictions = sorted(predictions, key=lambda x: x[0])
print(predictions)
with open(args.output, 'w', encoding='utf-8') as wf:
    for one in predictions:
        wf.write('{}\n'.format(one[1]))
