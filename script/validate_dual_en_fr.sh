#!/bin/sh

nematus=/home/minhquang.pham/nematus-master

# path to 
scorer=/home/minhquang.pham/Dual_NMT/script/multi-bleu.perl

# theano device
device=cpu

#index

index=$1
data_set=$2
# model prefix
prefix=/home/minhquang.pham/Dual_NMT/models/dual$index/model_dual.npz.en_fr

dev=/home/minhquang.pham/Dual_NMT/data/validation/hit/hit.en.tok.shuf.dev.tok
ref=/home/minhquang.pham/Dual_NMT/data/validation/hit/hit.fr.tok.shuf.dev.tok

if [ "$data_set" = "concatenated" ]; then
dev=/home/minhquang.pham/Dual_NMT/data/received/concatenated/concatenated.en.tok.shuf.test.tok.shuf.dev.tok
ref=/home/minhquang.pham/Dual_NMT/data/received/concatenated/concatenated.fr.tok.shuf.test.tok.shuf.dev.tok
fi
# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m $prefix.dev.npz \
     -i $dev \
     -o $dev.output.en_fr_$index.dev \
     -k 15 -n -p 3


#./postprocess-dev.sh < $dev.output.fr_en.dev > $dev.output.postprocessed.fr_en.dev


## get BLEU
BEST=`cat ${prefix}_best_bleu || echo 0`
$scorer $ref < $dev.output.en_fr_$index.dev >> ${prefix}_bleu_scores
BLEU=`$scorer $ref < $dev.output.en_fr_$index.dev | cut -f 3 -d ' ' | cut -f 1 -d ','`
BETTER=`echo "$BLEU > $BEST" | bc`

echo "BLEU = $BLEU"

# save model with highest BLEU
if [ "$BETTER" = "1" ]; then
  echo "new best; saving"
  echo $BLEU > ${prefix}_best_bleu
  cp ${prefix}.dev.npz ${prefix}.npz.best_bleu
fi

DIR=/home/minhquang.pham/Dual_NMT/data/validation/hit/translate/dev_$index

if [ $data_set = 'concatenated' ]; then
DIR=/home/minhquang.pham/Dual_NMT/data/received/concatenated/translate/dev_$index
fi

if [ ! -d "$DIR" ]; then
echo $DIR
mkdir $DIR
fi

mv $dev.output.en_fr_$index.dev $DIR
