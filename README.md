# Dual_Learning
This is a project that I did in my last internship. The paper can be found at https://arxiv.org/abs/1611.00179 . I have implemented it with small copora (50000 sentences) and I got good results. I am currently working on this, doing more experiments with large corpora (millions sentences) and gradually improving the clarity of the code. I hope that I can finish it soon. There is a lot of work whereas I am still busy at school. Thanks for your interests. <br />
Todo: <br />
I suggest you using my new version implementation of Dual Learning which uses Q-value to estimate Policy Gradient. <br />
To train model, you need to: <br />
+) pretrain 2 language models (using code of dl4mt) and put them in models/LM  <br />
+) pretrain 2 translation models using old version of nematus provided in my Github (for warmstart) and put them in models/name_of_your_warmstart_directory/NMT. <br />
+) train bilingual word embeddings for 2 languages ( I suggest using MUSE library of FAIR) and put them in data/word_emb/name_of_embedding_directory. <br />
(You should remember that Dual Learning is only used to improve performance of pretrained models by using monolingual corpora.)<br />

