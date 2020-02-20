import sys
sys.path.extend(["../../","../","./"])
import random
import argparse
from driver.Config import *
from data.Dataloader import *
from modules.BertModel import *
from modules.BertTokenHelper import *
import pickle
from modules.WordLSTM import *
from modules.PretrainedWordEncoder import *
from modules.Decoder import *
from modules.EDUSegmenter import *
from allennlp.modules.conditional_random_field import ConditionalRandomField as CRF
from driver.TrainTest import predict
from driver.TrainTest import scripts_evaluate

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
    argparser.add_argument('--config_file', default='rst_eduseg_model/config.cfg')
    argparser.add_argument('--model', default='rst_eduseg_model/model.13')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--test_file', default='')
    argparser.add_argument('--eval', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    test_data = read_corpus(args.test_file)

    vocab = pickle.load(open(config.load_vocab_path, 'rb'))
    seg_model = torch.load(args.model)

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
    crf = CRF(num_tags=vocab.seglabel_size,
              constraints=None,
              include_start_end_transitions=False)

    pwordEnc.load_state_dict(seg_model['pwordEnc'])
    wordLSTM.load_state_dict(seg_model['wordLSTM'])
    dec.load_state_dict(seg_model['dec'])
    crf.load_state_dict(seg_model['crf'])

    if config.use_cuda:
        wordLSTM.cuda()
        pwordEnc.cuda()
        dec.cuda()
        crf.cuda()

    segmenter = EDUSegmenter(pwordEnc, wordLSTM, dec, crf, config)
    predict(test_data, segmenter, vocab, config, tok, args.test_file + '.out')

    if args.eval:
        scripts_evaluate(config, args.test_file, args.test_file + '.out')

