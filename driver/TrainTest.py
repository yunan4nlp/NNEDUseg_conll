import sys
sys.path.extend(["../../","../","./"])
import random
import itertools
import argparse
from driver.Config import *
from data.Dataloader import *
from data.Vocab import *
from modules.BertModel import *
from modules.BertTokenHelper import *
import pickle
from modules.WordLSTM import *
from modules.PretrainedWordEncoder import *
from modules.Decoder import *
from modules.EDUSegmenter import *
import time

class Optimizer:
    def __init__(self, parameter, config, lr):
        self.optim = torch.optim.Adam(parameter, lr=lr, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon, weight_decay=config.l2_reg)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()
    def schedule(self):
        self.scheduler.step()
    def zero_grad(self):
        self.optim.zero_grad()
    @property
    def lr(self):
        return self.scheduler.get_lr()

def train(train_data, dev_data, test_data, segmenter, vocab, config, tokenizer):
    train_insts = inst(train_data)

    model_param = filter(lambda p: p.requires_grad,
                         itertools.chain(
                             segmenter.pwordEnc.parameters(),
                             segmenter.wordLSTM.parameters(),
                             segmenter.dec.parameters(),
                         )
                         )

    model_optimizer = Optimizer(model_param, config, config.learning_rate)

    global_step = 0
    best_FF = 0
    batch_num = int(np.ceil(len(train_insts) / float(config.train_batch_size)))

    for iter in range(config.train_iters):
        start_time = time.time()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        overall_label_correct,  overall_total_label = 0, 0
        for onebatch in data_iter(train_insts, config.train_batch_size, True):
            bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask, word_mask, label_mask = \
                batch_pretrain_variable_sent_level(onebatch, vocab, config, tokenizer)
            batch_gold_indexs = batch_variable_label(onebatch, vocab)

            segmenter.train()
            #with torch.autograd.profiler.profile() as prof:
            segmenter.encode(bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask,
                             word_mask)

            batch_predict_indexs = segmenter.decode(label_mask)

            loss = segmenter.compute_loss(batch_gold_indexs, label_mask)
            loss = loss / config.update_every
            loss_value = loss.data.cpu().numpy()
            loss.backward()

            total_labels, correct_labels = segmenter.compute_accuracy(batch_predict_indexs, batch_gold_indexs)

            overall_total_label += total_labels
            overall_label_correct += correct_labels
            during_time = float(time.time() - start_time)
            acc = overall_label_correct / overall_total_label
            #acc = 0
            print("Step:%d, Iter:%d, batch:%d, time:%.2f, acc:%.2f, loss:%.2f"
                  %(global_step, iter, batch_iter,  during_time, acc, loss_value))
            batch_iter += 1

            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(model_param, max_norm=config.clip)
                model_optimizer.step()
                model_optimizer.zero_grad()

                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                print("Dev:")
                predict(dev_data, segmenter, vocab, config, tokenizer, config.dev_file + '.' + str(global_step))

                dev_FF = scripts_evaluate(config, config.dev_file, config.dev_file + '.' + str(global_step))

                #dev_FF = evaluate(config.dev_file, config.dev_file + '.' + str(global_step))

                print("Test:")
                predict(test_data, segmenter, vocab, config, tokenizer, config.test_file + '.' + str(global_step))
                scripts_evaluate(config, config.test_file, config.test_file + '.' + str(global_step))

                if dev_FF > best_FF:
                    print("Exceed best Full F-score: history = %.2f, current = %.2f" % (best_FF, dev_FF))
                    best_FF = dev_FF
                    if config.save_after >= 0 and iter >= config.save_after:
                        segmenter_model = {
                            "pwordEnc": segmenter.pwordEnc.state_dict(),
                            "wordLSTM": segmenter.wordLSTM.state_dict(),
                            "dec": segmenter.dec.state_dict(),
                            }
                        torch.save(segmenter_model, config.save_model_path + "." + str(global_step))
                        print('Saving model to ', config.save_model_path + "." + str(global_step))

def scripts_evaluate(config, gold_file, predict_file):
    cmd = "python %s %s %s" % (config.eval_scripts, gold_file, predict_file)
    F_exec = os.popen(cmd).read()
    info = F_exec.strip().split("\n")
    fscore = info[-1].split(': ')[-1]
    print(' '.join(info))
    return float(fscore)

def evaluate(gold_file, predict_file):
    gold_data = read_corpus(gold_file, True)
    predict_data = read_corpus(predict_file, True)
    seg_metric = Metric()
    for gold_doc, predict_doc in zip(gold_data, predict_data):
        gold_edus = gold_doc.extract_EDUstr()
        predict_edus = predict_doc.extract_EDUstr()
        seg_metric.overall_label_count += len(gold_edus)
        seg_metric.predicated_label_count += len(predict_edus)
        seg_metric.correct_label_count += len(set(gold_edus) & set(predict_edus))
    print("edu seg:", end=" ")
    seg_metric.print()
    return seg_metric.getAccuracy()

def predict(data, segmenter, vocab, config, tokenizer, outputFile):
    start = time.time()
    segmenter.eval()
    outf = open(outputFile, mode='w', encoding='utf8')
    for docbatch in data_iter(data, 10, False):
        predict_insts = inst(docbatch)
        batch_predict_indexs = []
        for sentbatch in data_iter(predict_insts, config.test_batch_size, False):
            bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask, word_mask, label_mask = \
                batch_pretrain_variable_sent_level(sentbatch, vocab, config, tokenizer)
            # with torch.autograd.profiler.profile() as prof:
            segmenter.encode(bert_indice_input, bert_segments_ids, bert_piece_ids, bert_mask,
                         word_mask)
            batch_predict_indexs += segmenter.decode(label_mask)
        offset = 0
        batch_size = len(docbatch)
        for idx in range(batch_size):
            doc = docbatch[idx]
            outf.write(doc.firstline + '\n')
            for sentence_conll in doc.sentences_conll:
                predict_indexs = batch_predict_indexs[offset]
                assert len(predict_indexs) == len(sentence_conll)
                predict_labels = vocab.id2seglabel(predict_indexs)
                for idx, line in enumerate(sentence_conll):
                    if predict_labels[idx] == 'b' or idx == 0:
                        predict_label = 'BeginSeg=Yes'
                    elif predict_labels[idx] == 'i':
                        predict_label = '_'
                    info = line.strip().split("\t")
                    info[9] = predict_label
                    predict_line = '\t'.join(info)
                    outf.write(predict_line + '\n')
                offset += 1
                outf.write('\n')
    outf.close()
    end = time.time()
    during_time = float(end - start)
    print("doc num: %d, segment time = %.2f " % (len(data), during_time))

def parse_sentence_seg(sentences_seg, vocab):
    sentences_labels = []
    for sentence_seg in sentences_seg:
        sent_label = vocab.id2seglabel(sentence_seg)
        sent_label[0] = 'b'
        sentences_labels += sent_label
    start = 0
    EDUs_id = []
    for idx, label in enumerate(sentences_labels):
        if idx + 1 < len(sentences_labels) and sentences_labels[idx + 1] == 'b':
            end = idx
            EDUs_id.append([start, end])
            start = end + 1
        elif idx + 1 == len(sentences_labels):
            end = idx
            EDUs_id.append([start, end])
            start = end + 1
    return EDUs_id

if __name__ == '__main__':
    ### process id
    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))

    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='examples/default.cfg')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    train_data = read_corpus(config.train_file)
    dev_data = read_corpus(config.dev_file)
    test_data = read_corpus(config.test_file)

    vocab = creatVocab(train_data, config.min_occur_count)

    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    print('Load pretrained encoder.....')
    tok = BertTokenHelper(config.bert_dir)
    enc_model = BertExtractor(config)
    print(enc_model)
    print('Load pretrained encoder ok')

    pwordEnc = PretrainedWordEncoder(config, enc_model, enc_model.bert_hidden_size, enc_model.layer_num)
    wordLSTM = WordLSTM(vocab, config)
    dec = Decoder(vocab, config)
    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    if config.use_cuda:
        wordLSTM.cuda()
        pwordEnc.cuda()
        dec.cuda()

    segmenter = EDUSegmenter(pwordEnc, wordLSTM, dec, config)
    train(train_data, dev_data, test_data, segmenter, vocab, config, tok)
