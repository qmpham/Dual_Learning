home_dir = "/users/limsi_nmt/minhquang/"
from Dual_training_1 import train
if __name__ == '__main__':
    validerr = train(
              optimizers_ = None,
              saveto = home_dir + 'Dual_NMT/models/dual50/model_dual.npz',
              clip_c = 1.,
              n_words_en = 16000,  # english vocabulary size
              n_words_fr = 19000,  # french vocabulary size
              reload_ = False,
	      sampling = "beam_search",
              numb_samplings = 2,
              batch_size= 30,
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
              
              args_en_fr_1 = "50",
              args_en_fr_2 = "concatenated",
              args_fr_en_1 = "50",
              args_fr_en_2 = "concatenated",
              reward_scale = 1.0,
              alpha = 0.9,

              valid_en = home_dir + "Dual_NMT/data/received/concatenated/concatenated.en.tok.shuf.test.tok.shuf.dev.tok",
              valid_fr = home_dir + "Dual_NMT/data/received/concatenated/concatenated.fr.tok.shuf.test.tok.shuf.dev.tok",
              dataset_bi_en = home_dir + "Dual_NMT/data/received/concatenated/concatenated.en.tok.shuf.train.tok",
              dataset_bi_fr = home_dir + "Dual_NMT/data/received/concatenated/concatenated.fr.tok.shuf.train.tok",
              dataset_mono_en = home_dir + "Dual_NMT/data/received/concatenated/concatenated.en.tok.shuf.dual.tok",
              dataset_mono_fr = home_dir + "Dual_NMT/data/received/concatenated/concatenated.fr.tok.shuf.dual.tok",
              vocab_en = home_dir + "Dual_NMT/data/received/concatenated/concatenated.en.tok.json",
              vocab_fr = home_dir + "Dual_NMT/data/received/concatenated/concatenated.fr.tok.json",
              test_en = home_dir + "Dual_NMT/data/received/concatenated/concatenated.en.tok.shuf.test.tok.shuf.test.tok",
              test_fr = home_dir + "Dual_NMT/data/received/concatenated/concatenated.fr.tok.shuf.test.tok.shuf.test.tok",
              path_trans_en_fr = home_dir + "Dual_NMT/models/NMT/lowercased_warm_start_5/model_en_fr.npz.npz.best_bleu",
              path_trans_fr_en = home_dir + "Dual_NMT/models/NMT/lowercased_warm_start_5/model_fr_en.npz.npz.best_bleu",
              path_mono_en = home_dir + "Dual_NMT/models/LM/lowercased_2/model_lm_en.npz",
              path_mono_fr = home_dir + "Dual_NMT/models/LM/lowercased_2/model_lm_fr.npz",
              word_emb_en_path = home_dir + "Dual_NMT/data/word_emb_2/envec.txt",
              word_emb_fr_path = home_dir + "Dual_NMT/data/word_emb_2/frvec.txt",

              external_validation_script_en_fr = home_dir + "script/validate_dual_en_fr_shared_2.sh",
              external_validation_script_fr_en = home_dir + "script/validate_dual_fr_en_shared_2.sh", 
                           
              )
    print validerr

