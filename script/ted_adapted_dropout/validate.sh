#!/bin/sh

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/people/burlot/env/nematus

# path to 
scorer=/people/allauzen/dev/cpp/ncode/script/multi-bleu.perl

# theano device
device=cpu

# model prefix
prefix=model.npz

dev=data/dev.bpe.en
ref=data/dev10-11.en2cs.cs.tok

# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m $prefix.dev.npz \
     -i $dev \
     -o $dev.output.dev \
     -k 12 -n -p 1


./postprocess-dev.sh < $dev.output.dev > $dev.output.postprocessed.dev


## get BLEU
BEST=`cat ${prefix}_best_bleu || echo 0`
$scorer $ref < $dev.output.postprocessed.dev >> ${prefix}_bleu_scores
BLEU=`$scorer $ref < $dev.output.postprocessed.dev | cut -f 3 -d ' ' | cut -f 1 -d ','`
BETTER=`echo "$BLEU > $BEST" | bc`

echo "BLEU = $BLEU"

# save model with highest BLEU
if [ "$BETTER" = "1" ]; then
  echo "new best; saving"
  echo $BLEU > ${prefix}_best_bleu
  cp ${prefix}.dev.npz ${prefix}.npz.best_bleu
fi
