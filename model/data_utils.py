import numpy as np
import os

from nltk import word_tokenize

# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
        ERROR: Unable to locate file {}.
        FIX: Have you tried running python build_data.py first?
        This will build vocab file from your train, test and dev sets and
        trimm your word vectors.""".format(filename)
        super(MyIOError, self).__init__(message)


class Dataset(object):
    """Class that iterates.

    __iter__ method yields a tuple (tokens, tags) -> one sample
        tokens: list of raw token
        tags: list of raw tag:
        for example: line(0-19) -> tokens -> one sample;
        -----------------------------------------
        0: token0, sep, tag0
        1: token1, sep, tag1
        ...
        20: (blank line)
        21: token2, sep, tag2
        22: token3, sep, tag3
        -----------------------------------------
    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = Dataset(filename)
        for sentence, tags in data:
            pass
        ```

    """

    def __init__(self, filename, processing_word=None, processing_tag=None, max_iter=None, sep="\t", tokenLevel=1):
        """
        Args:
            filename: path to the file
            processing_word: (optional) function that takes a word as input
            processing_tag: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield
            tokenLevel: input token level: 0-chars; 1-words; 2-sentences;

        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.max_iter = max_iter
        self.length = None
        self.sep = sep
        self.tokenLevel = tokenLevel

    def __iter__(self):
        """类的迭代器可以使用for循环迭代类.

        Return: words, tags:
            - tokenLevel=0: words=[((1, 43, 78), 425), ...], tags=[B_PER, ...]
            - tokenLevel=1: words=[425, 498, ...],           tags=[B_PER, ...]
            - tokenLevel=2: words=[[425, 498], ...],         tags=[B_PER, ...]

        """
        niter = 0
        with open(self.filename) as f:
            tokens, tags = [], []
            numToken = 0
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    numToken = 0
                    # 空行分隔， tokens -> sequence
                    if len(tokens) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            # 超过最大迭代次数，break
                            break
                        yield tokens, tags
                        tokens, tags = [], []
                else:
                    numToken += 1
                    ls = line.split(self.sep)
                    if len(ls) != 2:
                        ls = line.split()
                    token, tag = ls[0], ls[1]
                    if self.processing_word is not None:
                        # return list of word id
                        token = self.processing_word(token, numToken)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag, -1)
                    tokens += [token]
                    tags += [tag]

    def __len__(self):
        """Iterates once over the corpus to set and store length when call len()."""
        if self.length is None:
            # sample length: 即分隔的空行数
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocabs(datasets):
    """Build vocabulary from an iterable of datasets objects

    Args:
        datasets: a list of dataset objects

    Returns:
        a set of all the words in the dataset

    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    for dataset in datasets:
        for words, tags in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags


def get_char_vocab(dataset):
    """Build char vocabulary from an iterable of datasets objects

    Args:
        dataset: a iterator yielding tuples (sentence, tags)

    Returns:
        a set of all the characters in the dataset

    """
    vocab_char = set()
    for words, _ in dataset:
        for word in words:
            vocab_char.update(word)

    return vocab_char


def get_glove_vocab(filename):
    """Load vocab from file

    Args:
        filename: path to the glove vectors

    Returns:
        vocab: set() of strings
    """
    print("Building vocab...")
    vocab = set()
    with open(filename) as f:
        for line in f:
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one word per line.

    Returns:
        d: dict[word] = index

    """
    try:
        d = dict()
        with open(filename) as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx

    except IOError:
        raise MyIOError(filename)
    return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings

    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def get_processing_word(vocab_words=None, vocab_chars=None, lowercase=False, tokenLevel=1, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab_words: dict[word] = idx
        vocab_chars: dict[char] = idx
        lowercase: lower case
        tokenLevel: input token level: 0-chars; 1-words; 2-sentences;
        allow_unk: allow unknown word

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = word id
                 = (list of char ids, word id)
                 = (list of word ids, sentence relative location id)

    """

    # map function
    def f(token, idt):
        # 0. get chars of words
        if vocab_chars is not None and tokenLevel == 0:
            char_ids = []
            for char in token:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]
        # 1. get words of sentence
        elif tokenLevel == 2:
            sentence_ids = []
            for word in word_tokenize(token):
                # ignore chars out of vocabulary
                if word in vocab_words:
                    sentence_ids += [vocab_words[word]]
                else:
                    if allow_unk:
                        sentence_ids += [vocab_words[UNK]]
                    else:
                        raise Exception("Unknow key is not allowed. Check that your vocab (tags?) is correct")

        # 2. preprocess token
        if lowercase:
            token = token.lower()
        if token.isdigit():
            token = NUM

        # 2. get id of word
        if vocab_words is not None:
            if token in vocab_words:
                token = vocab_words[token]
            else:
                if allow_unk:
                    token = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        if vocab_chars is not None and tokenLevel == 0:
            # (list of char ids, word id)
            return char_ids, token
        elif tokenLevel == 2:
            # (list of word ids, UNK id)
            return sentence_ids, idt
        else:
            # word id
            return token

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 0:
        # char embedding
        max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    elif nlevels == 2:
        # sentence embedding
        max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    else:
        # word embedding
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (samples, tags) tuples
        minibatch_size: (int)

    Yields:
        list of tuples

    """
    x_batch, y_batch = [], []
    for (x, y) in data:
        if len(x_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []

        if type(x[0]) == tuple:
            # char embedding, sentence embedding
            x = zip(*x)
        x_batch += [x]
        y_batch += [y]

    if len(x_batch) != 0:
        yield x_batch, y_batch


def get_chunk_type(tok, idx_to_tag, hypen="_"):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}

    Returns:
        tuple: "B", "PER"

    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split(hypen)[0]
    tag_type = tag_name.split(hypen)[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """Given a sequence of tags, group entities and their position

    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4

    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]

    """
    default = tags[NONE]
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks
