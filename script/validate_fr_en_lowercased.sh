#!/bin/sh

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/users/limsi_nmt/minhquang/nematus-master

# path to 
scorer=/users/limsi_nmt/minhquang/script/multi-bleu.perl

# theano device
device=cpu

# model prefix
prefix=/users/limsi_nmt/minhquang/nematus-master/models/model_lowercased_with_prior/model_fr_en/model_fr_en.npz

dev=/users/limsi_nmt/minhquang/Dual_NMT/data/validation/hit/hit.fr.tok.shuf.dev.tok
ref=/users/limsi_nmt/minhquang/Dual_NMT/data/validation/hit/hit.en.tok.shuf.dev.tok

# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m $prefix.dev.npz \
     -i $dev \
     -o $dev.output.lowercased_with_prior.dev \
     -k 12 -n -p 5

## get BLEU
BEST=`cat ${prefix}_best_bleu || echo 0`
$scorer $ref < $dev.output.lowercased_with_prior.dev >> ${prefix}_bleu_scores
BLEU=`$scorer $ref < $dev.output.lowercased_with_prior.dev | cut -f 3 -d ' ' | cut -f 1 -d ','`
BETTER=`echo "$BLEU > $BEST" | bc`

echo "BLEU = $BLEU"

# save model with highest BLEU
if [ "$BETTER" = "1" ]; then
  echo "new best; saving"
  echo $BLEU > ${prefix}_best_bleu
  cp ${prefix}.dev.npz ${prefix}.npz.best_bleu
fi

DIR=/users/limsi_nmt/minhquang/Dual_NMT/data/validation/hit/translate/lowercased_with_prior_dev
if [ ! -d "$DIR" ]; then
echo $DIR
mkdir $DIR
fi

mv $dev.output.lowercased_with_prior.dev $DIR

