# theano device
device=cpu
## 
#nematus=/people/allauzen/dev/rico/nematus/nematus
nematus=/people/burlot/env/nematus/nematus

## AApython stufs
MYPYTHON=/people/allauzen/venv/kevin/bin
MYPYLIB=/people/allauzen/venv/kevin/lib # python2.7/site-packages
export PATH=$MYPYTHON:${PATH}
export PYTHONPATH=$nematus:$MYPYLIB:$MYPYLIB/python2.7:$MYPYLIB/python2.7/site-packages
which python
python -c 'import numpy; print "numpy OK"; import theano; print "theano OK"'

## 
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python -u  ./config.py
