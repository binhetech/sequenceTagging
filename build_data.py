from model.config import Config
from model.data_utils import Dataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word


def main():
    """Procedure to build data

    You MUST RUN this procedure. It iterates over the whole dataset (train,
    dev and test) and extract the vocabularies in terms of words, tags, and
    characters. Having built the vocabularies it writes them in a file. The
    writing of vocabulary in a file assigns an id (the line #) to each word.
    It then extract the relevant GloVe vectors and stores them in a np array
    such that the i-th entry corresponds to the i-th word in the vocabulary.


    Args:
        config: (instance of Config) has attributes like hyper-params...

    """
    # get config and processing of words
    config = Config(load=False)
    processing_word = get_processing_word(lowercase=True)

    # Generators
    train = Dataset(config.filename_train, processing_word, sep=config.sep, tokenLevel=config.tokenLevel)
    dev = Dataset(config.filename_dev, processing_word, sep=config.sep, tokenLevel=config.tokenLevel)
    test = Dataset(config.filename_test, processing_word, sep=config.sep, tokenLevel=config.tokenLevel)

    # Build Word and Tag vocab
    vocab_words, vocab_tags = get_vocabs([train, dev, test])
    vocab_glove = get_glove_vocab(config.filename_glove)

    vocab = vocab_words & vocab_glove
    vocab.add(UNK)
    vocab.add(NUM)

    # Save vocab
    write_vocab(vocab, config.filename_words, "words")
    write_vocab(vocab_tags, config.filename_tags, "tags")

    # Trim GloVe Vectors
    vocab = load_vocab(config.filename_words)
    export_trimmed_glove_vectors(vocab, config.filename_glove, config.filename_trimmed, config.dim_word)

    # Build and save char vocab
    train = Dataset(config.filename_train, sep=config.sep, tokenLevel=config.tokenLevel)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.filename_chars, "chars")


if __name__ == "__main__":
    main()
