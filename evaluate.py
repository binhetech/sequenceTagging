from model.data_utils import Dataset
from model.ner_model import TaggingModel
from model.config import Config
from nltk import sent_tokenize


def align_data(data):
    """Given dict with lists, creates aligned strings

    Adapted from Assignment 3 of CS224N

    Args:
        data: (dict) data["x"] = ["I", "love", "you"]
              (dict) data["y"] = ["O", "O", "O"]

    Returns:
        data_aligned: (dict) data_align["x"] = "I love you"
                           data_align["y"] = "O O    O  "

    """
    spacings = [max([len(seq[i]) for seq in data.values()])
                for i in range(len(data[list(data.keys())[0]]))]
    data_aligned = dict()

    # for each entry, create aligned string
    for key, seq in data.items():
        str_aligned = ""
        for token, spacing in zip(seq, spacings):
            str_aligned += token + " " * (spacing - len(token) + 1)

        data_aligned[key] = str_aligned

    return data_aligned


def interactive_shell(model, config):
    """Creates interactive shell to play with model

    Args:
        model: instance of NERModel

    """
    model.logger.info("""
This is an interactive mode.
To exit, enter 'exit'.
You can enter a sentence like
input> I love Paris""")

    while True:
        sentence = input("input> ")
        if config.tokenLevel == 2:
            # sentence
            tokensRaw = sent_tokenize(sentence.strip())
        else:
            # word
            tokensRaw = sentence.strip().split()

        if tokensRaw == ["exit"]:
            break

        preds = model.predict(tokensRaw)
        to_print = align_data({"input": tokensRaw, "output": preds})

        for key, seq in to_print.items():
            model.logger.info(seq)


def main():
    # create instance of config
    config = Config()

    # build model
    model = TaggingModel(config)
    model.build()
    model.restore_session(config.dir_model)

    # create dataset
    test = Dataset(config.filename_test, config.processing_word, config.processing_tag, config.max_iter, sep=config.sep,
                   tokenLevel=config.tokenLevel)

    # evaluate and interact
    model.evaluate(test)
    interactive_shell(model, config)


if __name__ == "__main__":
    main()
