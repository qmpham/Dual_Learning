index=$1
device=$2
nematus=~/nematus-master
source ~/anaconda3/bin/activate nmt_env
MYPYTHON=~/anaconda3/envs/nmt_env/bin/
MYPYLIB=~/anaconda3/envs/nmt_env/lib/
export PATH=$MYPYTHON:${PATH}
export PYTHONPATH=$MYPYLIB:$nematus:$MYPYLIB/python2.7:$MYPYLIB/python2.7/dists-packages
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
export CUDA_VISIBLE_DEVICES=$device

DIR=~/Dual_NMT/models/dual$index
if [ ! -d "$DIR" ]; then
echo $DIR
mkdir $DIR
fi

which python
python -c 'import numpy; print "numpy OK"; import theano; print "theano OK"; import nematus; print "nematus OK"'
THEANO_FLAGS='mode=FAST_RUN,optimizer=fast_run,device=gpu,floatX=float32,on_unused_input=warn' python -u /home/minhquang.pham/Dual_NMT/config_Q_value_$index.py
