#!/bin/sh

# this sample script translates a test set, including
# preprocessing (tokenization, truecasing, and subword segmentation),
# and postprocessing (merging subword units, detruecasing, detokenization).

# instructions: set paths to mosesdecoder, subword_nmt, and nematus,
# the run "./translate.sh < input_file > output_file"

# suffix of source language
SRC=en

# suffix of target language
TRG=cs

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/vol/soft-tlp/translate/moses/3.0/m15x/

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=/people/burlot/env/nematus/subword-nmt

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/people/burlot/env/nematus

# theano device
device=cpu

# preprocess
#$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC | \
#$mosesdecoder/scripts/tokenizer/tokenizer.perl -l $SRC -a | \
#$mosesdecoder/scripts/recaser/truecase.perl -model truecase-model.$SRC | \
$subword_nmt/apply_bpe.py -c $SRC$TRG.bpe | \
# translate
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m model.npz \
     -k 12 -n -p 1 | \
# postprocess
sed 's/\@\@ //g'
#$mosesdecoder/scripts/recaser/detruecase.perl | \
#$mosesdecoder/scripts/tokenizer/detokenizer.perl -l $TRG
