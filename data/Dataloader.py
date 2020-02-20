import numpy as np
import torch
from torch.autograd import Variable
from data.Doc import *


def read_sent(inf):
    sentence = []
    for line in inf:
        line = line.strip()
        if line == '':
            yield sentence
            sentence = []
        else:
            sentence.append(line)
    if len(sentence) > 0:
        yield sentence

def read_corpus(file_path, eval=False):
    with open(file_path, mode='r', encoding='utf8') as inf:
        doc_data = []
        for inst in read_sent(inf):
            if inst[0].find("# newdoc id =") == 0:
                doc_name = inst[0].split('=')[1].strip()
                doc = Doc()
                doc.firstline = inst[0]
                doc.name = doc_name
                doc.sentences_conll.append(inst[1:])
                doc_data.append(doc)
            else:
                doc.sentences_conll.append(inst)
    doc_num = len(doc_data)
    sent_num = 0
    for doc in doc_data:
        sent_num += len(doc.sentences_conll)
        doc.extract_conll()
    if not eval:
        print("Info: ", file_path)
        print("Doc num: ", doc_num)
        print("Sentence num: ", sent_num)
    return doc_data

def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield sentences



def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch

def batch_variable_label(batch, vocab):
    max_sent_len = max([len(data[0]) for data in batch])
    gold_labels_id = []
    for idx, data in enumerate(batch):
        labels = data[1][:max_sent_len]
        label_ids = vocab.seglabel2id(labels)
        label_index = np.ones(len(label_ids), dtype=np.int32) * -1
        for idx, index in enumerate(label_ids):
            label_index[idx] = index
        gold_labels_id.append(label_index)
    return gold_labels_id

def batch_pretrain_variable_sent_level(batch, vocab, config, tokenizer):
    batch_size = len(batch)
    max_bert_len = -1
    max_sent_len = max([len(data[0]) for data in batch])
    batch_bert_indices = []
    batch_segments_ids = []
    batch_piece_ids = []
    for data in batch:
        sent = data[0][:max_sent_len]
        bert_indice, segments_id, piece_id = tokenizer.bert_ids(' '.join(sent))
        batch_bert_indices.append(bert_indice)
        batch_segments_ids.append(segments_id)
        batch_piece_ids.append(piece_id)
        assert len(piece_id) == len(sent)
        assert len(bert_indice) == len(segments_id)
        bert_len = len(bert_indice)
        if bert_len > max_bert_len: max_bert_len = bert_len
    bert_indice_input = np.zeros((batch_size, max_bert_len), dtype=int)
    bert_mask = np.zeros((batch_size, max_bert_len), dtype=int)
    bert_segments_ids = np.zeros((batch_size, max_bert_len), dtype=int)
    bert_piece_ids = np.zeros((batch_size, max_sent_len, max_bert_len), dtype=float)

    word_mask = np.zeros((batch_size, max_sent_len), dtype=int)

    for idx in range(batch_size):
        bert_indice = batch_bert_indices[idx]
        segments_id = batch_segments_ids[idx]
        piece_id = batch_piece_ids[idx]

        bert_len = len(bert_indice)
        sent_len = len(piece_id)
        assert sent_len <= bert_len
        for idz in range(bert_len):
            bert_indice_input[idx, idz] = bert_indice[idz]
            bert_segments_ids[idx, idz] = segments_id[idz]
            bert_mask[idx, idz] = 1
        for idz in range(sent_len):
            for sid, piece in enumerate(piece_id):
                avg_score = 1.0 / (len(piece))
                for tid in piece:
                    bert_piece_ids[idx, sid, tid] = avg_score
        for idz in range(sent_len):
            word_mask[idx, idz] = 1

    bert_indice_input = torch.from_numpy(bert_indice_input)
    bert_segments_ids = torch.from_numpy(bert_segments_ids)
    bert_piece_ids = torch.from_numpy(bert_piece_ids).type(torch.FloatTensor)
    bert_mask = torch.from_numpy(bert_mask)

    word_mask = torch.from_numpy(word_mask).type(torch.FloatTensor)
    label_mask = word_mask.type(torch.LongTensor)

    return bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask, word_mask, label_mask

def inst(data):
    inst = []
    for doc in data:
        for sentence, sentence_labels in zip(doc.sentences, doc.sentences_labels):
            assert len(sentence) == len(sentence_labels)
            inst.append([sentence, sentence_labels])
    return inst
