import os

from .general_utils import get_logger
from .data_utils import get_trimmed_glove_vectors, load_vocab, get_processing_word


class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()

    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords = len(self.vocab_words)
        self.ntags = len(self.vocab_tags)
        self.nchars = len(self.vocab_chars)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words, self.vocab_chars, lowercase=True,
                                                   tokenLevel=self.tokenLevel)
        self.processing_tag = get_processing_word(self.vocab_tags, lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed) if self.use_pretrained else None)

    # general config
    dir_output = "results/test/"
    dir_model = dir_output + "model/"
    path_log = dir_output + "log.txt"

    # embeddings
    dim_word = 300
    dim_char = 100
    dim_sentence = 300

    # glove files
    filename_glove = "data/glove.840B/glove.840B.{}d.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    filename_trimmed = "data/glove.840B/glove.840B.{}d.trimmed.npz".format(dim_word)
    use_pretrained = True

    # dataset
    filename_train = "data/discourse/train-sent.csv"
    filename_test = "data/discourse/test-sent.csv"
    filename_dev = "data/discourse/dev-sent.csv"

    # filename_dev = filename_test = filename_train = "data/test.txt" # test

    max_iter = None  # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "data/words.txt"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"

    # training
    train_embeddings = False
    nepochs = 30
    batch_size = 64
    dropout = 0.5
    lr_method = "adam"
    lr = 0.001
    lr_decay = 0.9
    clip = -1  # if negative, no clipping
    nepoch_no_imprv = 5
    restore = False  # restore from saved session

    # model hyperparameters
    hidden_size_char = 100  # lstm on chars
    hidden_size_sentence = 300  # lstm on sentences
    hidden_size_lstm = 300  # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True  # if crf, training is 1.7x slower on CPU
    # tokenLevel: input token level: 0 - chars; 1 - words; 2 - sentences;
    tokenLevel = 2
    # each line token separator, "\t", "\n", ...
    sep = "\t"
