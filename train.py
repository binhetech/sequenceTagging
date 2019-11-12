from model.data_utils import Dataset
from model.tagging_model import TaggingModel
from model.config import Config


def main():
    # create instance of config
    config = Config()

    # build model
    model = TaggingModel(config)
    model.build()
    # init from stored weights
    if config.restore:
        model.restore_session(config.dir_model)  # optional, restore weights
        model.reinitialize_weights("proj")

    # create data sets
    dev = Dataset(config.filename_dev, config.processing_word, config.processing_tag, config.max_iter, sep=config.sep,
                  tokenLevel=config.tokenLevel)
    train = Dataset(config.filename_train, config.processing_word, config.processing_tag, config.max_iter,
                    sep=config.sep, tokenLevel=config.tokenLevel)

    # train model
    model.train(train, dev)


if __name__ == "__main__":
    main()
