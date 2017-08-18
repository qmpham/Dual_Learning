index=$1
device=$2
nematus=/users/limsi_nmt/minhquang/nematus-master
MYPYTHON=/usr/bin
MYPYLIB=/usr/lib
export PATH=$MYPYTHON:${PATH}
export PYTHONPATH=$nematus:$MYPYLIB:$MYPYLIB/python2.7:$MYPYLIB/python2.7/dists-packages

DIR=/users/limsi_nmt/minhquang/Dual_NMT/models/dual$index
if [ ! -d "$DIR" ]; then
echo $DIR
mkdir $DIR
fi

which python
python -c 'import numpy; print "numpy OK"; import theano; print "theano OK"; import nematus; print "nematus OK"'

THEANO_FLAGS="mode=FAST_RUN,device=$device,floatX=float32,on_unused_input=warn" python -u /users/limsi_nmt/minhquang/Dual_NMT/config_$index.py
