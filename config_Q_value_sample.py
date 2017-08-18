home_dir = "/users/limsi_nmt/minhquang/"
from Dual_training_Q_value import train
if __name__ == '__main__':
    validerr = train(
              optimizers_ = None,
              saveto = home_dir + 'Dual_NMT/models/dual80/model_dual.npz',
              clip_c = 1.,
              n_words_en = 15000,  # english vocabulary size
              n_words_fr = 15000,  # french vocabulary size
              reload_ = False,
	      sampling = "beam_search",
              numb_samplings = 2,
              batch_size= 60,
              valid_batch_size = 80,
              lrate_fw= 0.0001,
              lrate_bw = 0.001,
              dispFreq = 1000,
              validFreq = 10000,
              saveFreq = 10000,

              dist_scale = 0.0,
              length_constraint_scale = 0.0,

              use_second_update = True,
              using_word_emb_Bilbowa = True,
              
              args_en_fr_1 = "80",
              args_en_fr_2 = "hit",
              args_fr_en_1 = "80",
              args_fr_en_2 = "hit",
              reward_scale = 1.0,
              alpha = 0.9,
              
              path_trans_en_fr = home_dir + "Dual_NMT/models/NMT/lowercased_warm_start_1/model_en_fr.npz.npz.best_bleu",
              path_trans_fr_en = home_dir + "Dual_NMT/models/NMT/lowercased_warm_start_1/model_fr_en.npz.npz.best_bleu",

              dataset_bi_en = home_dir + "Dual_NMT/data/train/train10/train10.en.tok",
              dataset_bi_fr = home_dir + "Dual_NMT/data/train/train10/train10.fr.tok",


              external_validation_script_en_fr = home_dir + "script/validate_dual_en_fr_shared_2.sh",
              external_validation_script_fr_en = home_dir + "script/validate_dual_fr_en_shared_2.sh",              
              
              )
    print validerr

