import numpy
import os
import sys

VOCAB_SRC = 85000
VOCAB_TRG = 85000
SRC = "en"
TGT = "cs"
DATA_DIR = "data/"

from nematus.nmt import train
#from nmt import train


if __name__ == '__main__':
    validerr = train(saveto='./model.npz',
                    max_epochs=5,
                    reload_=True,
                    dim_word=500,
                    dim=1024,
                    n_words=VOCAB_TRG,
                    n_words_src=VOCAB_SRC,
                    decay_c=0.,
                    clip_c=1.,
                    lrate=0.0001,
                    optimizer='adam',
                    maxlen=50,
                    batch_size=80,
                    valid_batch_size=80,
                    datasets=[DATA_DIR + '/ted.bpe.' + SRC, DATA_DIR + '/ted.bpe.' + TGT],
                    valid_datasets=[DATA_DIR + '/dev.bpe.' + SRC, DATA_DIR + '/dev.bpe.' + TGT],
                    dictionaries=[DATA_DIR + '/vocab.' + SRC + '.json',DATA_DIR + '/vocab.' + TGT + '.json'],
                    validFreq=10000,
                    dispFreq=1000,
                    saveFreq=1000,
                    sampleFreq=10000,
                    use_dropout=True,
                    dropout_embedding=0.2, # dropout for input embeddings (0: no dropout)
                    dropout_hidden=0.2, # dropout for hidden layers (0: no dropout)
                    dropout_source=0, # dropout source words (0: no dropout)
                    dropout_target=0, # dropout target words (0: no dropout)
                    overwrite=False,
                    external_validation_script='./validate.sh')
    print validerr
