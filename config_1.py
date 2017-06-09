from Dual_training_2 import train
if __name__ == '__main__':
    validerr = train(
              optimizers_ = None,                          
              saveto = 'models/dual3/model_dual.npz', 
              clip_c = 1.,
              beam_search_size = 2,
              dispFreq = 100,
              validFreq = 2000,
              use_second_update = False,
              dataset_bi_en = "/people/minhquang/Dual_NMT/data/train/hit/hit.en.tok.shuf.train.tok",
              dataset_bi_fr = "/people/minhquang/Dual_NMT/data/train/hit/hit.en.tok.shuf.train.tok",             
              )
    print validerr