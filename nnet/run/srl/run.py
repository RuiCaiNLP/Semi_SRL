from nnet.run.runner import *
from nnet.ml.voc import *
from functools import partial
from nnet.nn_models.SRL import BiLSTMTagger


def make_local_voc(labels):
    return {i: label for i, label in enumerate(labels)}

def bio_reader(record):
    dbg_header, sent,  pos_tags, dep_parsing, root_dep_parsing, frame, target, f_lemmas, f_targets, labels_voc, \
    labels, all_l_ids, Predicate_link, Predicate_Labels_nd, Predicate_Labels = record.split('\t')
    labels_voc = labels_voc.split(' ')

    labels_voc.insert(0, '<pad>')
    frame = [frame] * len(labels_voc)
    words = []
    for word in sent.split(' '):
        words.append(word)

    pos_tags = pos_tags.split(' ')
    labels = labels.split(' ')
    all_l_ids = all_l_ids.split(' ')
    Predicate_link = Predicate_link.split(' ')
    Predicate_Labels_nd = Predicate_Labels_nd.split(' ')
    Predicate_Labels = Predicate_Labels.split(' ')

    if pos_tags[int(target)].startswith("V"):
        dbg_header = 'V'
    elif pos_tags[int(target)].startswith("N"):
        dbg_header = 'N'


    assert (len(words) == len(labels))

    # convert labels into indexes in labels_voc
    local_voc = {v: k for k, v in make_local_voc(labels_voc).items()}
    #labels = [local_voc[label] for label in labels]

    dep_parsing = dep_parsing.split()
    dep_parsing = [p.split('|') for p in dep_parsing]
    dep_parsing = [(p[0], int(p[1]), int(p[2])) for p in dep_parsing]

    root_dep_parsing = root_dep_parsing.split()
    root_dep_parsing = [p.split('|') for p in root_dep_parsing]
    root_dep_parsing = [(p[0], int(p[1]), int(p[2])) for p in root_dep_parsing]

    f_lemmas = f_lemmas.split(' ')
    f_targets = f_targets.split(' ')


    return dbg_header, words, pos_tags, dep_parsing, root_dep_parsing, frame, \
           np.int64(target), f_lemmas, np.int64(f_targets), labels_voc, labels, \
           all_l_ids, Predicate_link, Predicate_Labels_nd, Predicate_Labels


def unlabeled_reader(record):
    words = []
    for word in record.strip().split(' '):
        words.append(word)
    return words




class SRLRunner(Runner):
    def __init__(self):
        super(SRLRunner, self).__init__()

        self.word_voc = create_voc('file', self.a.word_voc)
        self.word_voc.add_unks()
        self.char_voc = create_voc('file', self.a.char_voc)
        self.char_voc.add_unks()
        self.freq_voc = frequency_voc(self.a.freq_voc)
        self.p_word_voc = create_voc('file', self.a.p_word_voc)
        self.p_word_voc.add_unks()
        self.role_voc = create_voc('file', self.a.role_voc)
        self.frame_voc = create_voc('file', self.a.frame_voc)
        self.pos_voc = create_voc('file', self.a.pos_voc)
        self.dep_voc = create_voc('file', self.a.dep_voc)
        self.specific_dep_voc = create_voc('file', self.a.specific_dep_voc)


        #log(self.word_voc.direct)
        log('SRLRunner has inistialized!')


    def add_special_args(self, parser):
        parser.add_argument(
            "--word-voc", required=True)
        parser.add_argument(
            "--char-voc", required=True)
        parser.add_argument(
            "--p-word-voc", required=True)
        parser.add_argument(
            "--freq-voc", required=True)
        parser.add_argument(
            "--role-voc", required=True)
        parser.add_argument(
            "--frame-voc", required=True)
        parser.add_argument(
            "--pos-voc", required=True
        )
        parser.add_argument(
            "--dep-voc", required=True
        )
        parser.add_argument(
            "--specific-dep-voc", required=True
        )
        parser.add_argument(
            "--word-embeddings", required=True
        )
        parser.add_argument(
            "--elmo-embeddings-0", required=False
        )
        parser.add_argument(
            "--elmo-embeddings-1", required=False
        )
        parser.add_argument(
            "--elmo-embeddings-2", required=False
        )
        parser.add_argument(
            "--data_partition", required=True
        )
        parser.add_argument(
            "--hps", help="model hyperparams", required=False
        )
        parser.add_argument(
            "--eval-dir", help="path to dir with eval data and scripts",
            required=True
        )


    def get_parser(self):
        return partial(bio_reader)

    def get_unlabeled(self):
        return partial(unlabeled_reader)

    def get_reader(self):
        return simple_reader

    def get_converter(self):
        def bio_converter(batch):
            header, sent_, pos_tags, dep_parsing, root_dep_parsing, frames, \
            targets, f_lemmas, f_targets, labels_voc, labels, all_l_ids, \
            Predicate_link, Predicate_Labels_nd, Predicate_Labels = list(zip(*batch))

            max_len = len(max(sent_, key=len))
            sent_padded_forchar = [["pad"]* max_len]*len(batch)
            for i in range(len(sent_)):
                for j in range(len(sent_[i])):
                    sent_padded_forchar[i][j] = sent_[i][j]

            max_char_len = 0
            for w in sent_:
                for c in w:
                    if len(list(c)) > max_char_len:
                        max_char_len = len(list(c))

            char = [[0 if c == "<pad>" else self.char_voc.vocalize(c.strip()) for c in w] for w in sent_padded_forchar]

            char_padded = np.zeros((len(batch), max_len, max_char_len))
            for i in range(len(sent_)):
                for j in range(max_len):
                    for k in range(len(char[i][j])):
                        char_padded[i, j, k] = char[i][j][k]

            char_padded = char_padded.astype('int64')
            sent = [self.word_voc.vocalize(w) for w in sent_]

            p_sent = [self.p_word_voc.vocalize(w) for w in sent_]

            pos_tags = [self.pos_voc.vocalize(w) for w in pos_tags]

            labels = [self.role_voc.vocalize(w) for w in labels]

            freq = [[self.freq_voc[self.word_voc.direct[i]] if
                     self.word_voc.direct[i] != '_UNK' else 0 for i in w] for
                    w
                    in sent]

            dep_seq = []
            for w in dep_parsing:
                tags = [p[0] for p in w]
                tags.insert(0, '<pad>')
                dep_seq.append(tags)
            dep_tags = [self.dep_voc.vocalize(p) for p in dep_seq]

            all_l_ids = [[int(r) for r in s ]for s in all_l_ids]
            Predicate_link = [[int(r) for r in s ]for s in Predicate_link]
            Predicate_Labels_nd = [self.dep_voc.vocalize(p) for p in Predicate_Labels_nd]
            Predicate_Labels = [self.specific_dep_voc.vocalize(p) for p in Predicate_Labels]

            dep_head = []
            for w in dep_parsing:
                heads = [int(p[2]) for p in w]
                heads.insert(0, -1)
                dep_head.append(heads)

            frames = [self.frame_voc.vocalize(f) for f in frames]
            labels_voc = [self.role_voc.vocalize(r) for r in labels_voc]
            lemmas_idx = [self.frame_voc.vocalize(f) for f in f_lemmas]


            sent_batch, sent_mask = mask_batch(sent)
            p_sent_batch, _ = mask_batch(p_sent)
            freq_batch, _ = mask_batch(freq)
            freq_batch = freq_batch.astype(dtype='float32')

            pos_batch, _ = mask_batch(pos_tags)
            dep_tag_batch, _ = mask_batch(dep_tags)


            all_l_ids_batch, _ = mask_batch(all_l_ids)
            Predicate_link_batch, _ = mask_batch(Predicate_link)
            Predicate_Labels, _ = mask_batch(Predicate_Labels)
            Predicate_Labels_nd, _ = mask_batch(Predicate_Labels_nd)

            dep_head_batch, _ = mask_batch(dep_head, for_Head=True)
            labels_voc_batch, labels_voc_mask = mask_batch(labels_voc)

            for line in labels_voc_mask:
                line[0] = 0

            labels_batch, _ = mask_batch(labels)
            frames_batch, _ = mask_batch(frames)


            region_mark = np.zeros(sent_batch.shape, dtype='int64')
            hps = eval(self.a.hps)
            rm = hps['rm']
            if rm >= 0:
                for r, row in enumerate(region_mark):
                    for c, column in enumerate(row):
                        if targets[r] - rm <= c <= targets[r] + rm:
                            region_mark[r][c] = 1

            sent_pred_lemmas_idx = np.zeros(sent_batch.shape, dtype='int64')
            for r, row in enumerate(sent_pred_lemmas_idx):
                for c, column in enumerate(row):
                    for t, tar in enumerate(f_targets[r]):
                        if tar == c:
                            sent_pred_lemmas_idx[r][c] = lemmas_idx[r][t]

            sent_pred_lemmas_idx = np.array(sent_pred_lemmas_idx, dtype='int64')

            assert (sent_batch.shape == sent_mask.shape)
            assert (frames_batch.shape == labels_voc_batch.shape == labels_voc_mask.shape)
            assert (labels_batch.shape == sent_batch.shape)


            return sent_batch, p_sent_batch, pos_batch, sent_mask, targets, frames_batch, \
                   labels_voc_batch, \
                   labels_voc_mask, freq_batch, \
                   region_mark, \
                   sent_pred_lemmas_idx, \
                   dep_tag_batch, dep_head_batch, labels_batch, all_l_ids_batch, Predicate_link_batch,\
                   Predicate_Labels_nd,\
                   Predicate_Labels, char_padded
        return bio_converter


    def get_unlabeled_converter(self):
        def bio_unlabeled_converter(batch):



            p_sent = [self.p_word_voc.vocalize(w) for w in batch]

            max_len = len(max(p_sent, key=len))
            sent_padded_forchar = [["pad"] * max_len] * len(batch)
            for i in range(len(batch)):
                for j in range(len(batch[i])):
                    sent_padded_forchar[i][j] = batch[i][j]

            max_char_len = 0
            for w in batch:
                for c in w:
                    if len(list(c)) > max_char_len:
                        max_char_len = len(list(c))

            char = [[0 if c == "<pad>" else self.char_voc.vocalize(c.strip()) for c in w] for w in sent_padded_forchar]

            char_padded = np.zeros((len(batch), max_len, max_char_len))
            for i in range(len(batch)):
                for j in range(max_len):
                    for k in range(len(char[i][j])):
                        char_padded[i, j, k] = char[i][j][k]

            sent = [self.word_voc.vocalize(w) for w in batch]
            sent_batch, sent_mask = mask_batch(sent)
            p_sent_batch, _ = mask_batch(p_sent)

            return sent_batch, p_sent_batch, sent_mask, char_padded
        return bio_unlabeled_converter


    def load_model(self):
        log("start to build model ....")
        hps = eval(self.a.hps)

        hps['vframe'] = self.frame_voc.size()
        hps['vword'] = self.word_voc.size()
        hps['vchar'] = self.char_voc.size()
        hps['vbio'] = self.role_voc.size()
        hps['vpos'] = self.pos_voc.size()
        hps['vdep'] = self.dep_voc.size()
        hps['batch_size'] = self.a.batch
        hps['svdep'] = self.specific_dep_voc.size()
        ## do not use pre-trained embedding for now
        hps['word_embeddings'] = parse_word_embeddings(self.a.word_embeddings)
        #hps['elmo_embeddings_0'] = parse_word_embeddings(self.a.elmo_embeddings_0)
        #hps['elmo_embeddings_1'] = parse_word_embeddings(self.a.elmo_embeddings_1)
        hps['in_arcs'] = True
        hps['out_arcs'] = True
        torch.manual_seed(1)

        return BiLSTMTagger(hps)



if __name__ == '__main__':
    SRLRunner().run()
