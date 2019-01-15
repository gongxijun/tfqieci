# tfqieci
切词系统
生成初始词表：
                    /data0/xijun1/conda/bin/python python/train/process_anno_file.py  /data0/xijun1/tfqieci/data/ pre_chars_for_w2v.txt     
去除频率较低的词
             output/bin/word2vec_bin  -train pre_chars_for_w2v.txt -save-vocab pre_vocab.txt -min-count 3 
    将低频词替换成UNK
            /data0/xijun1/conda/bin/python  /data0/xijun1/tfqieci/python/train/replace_unk.py pre_vocab.txt pre_chars_for_w2v.txt chars_for_w2v.txt
          

      用生成的数据训练词向量  
            output/bin/word2vec_bin -train chars_for_w2v.txt -output vec.txt -size 200 -sample 1e-4 -negative 5 -hs 1 -binary 0 -iter 15
    5.生成BILstm +crf需要的训练预料 train.py , test.py
        /data0/xijun1/conda/bin/python python/train/generate_training.py  vec.txt  /data0/xijun1/tfqieci/data/ doc_all.txt
    6. 得到train.txt , test.txt
        /data0/xijun1/conda/bin/python python/train/filter_sentence.py doc_all.txt                      
 生成词汇表
            /data0/xijun1/conda/bin/python dump_vocab.py vec.txt models/bbasic_vocab.txt 
      8.开始训练
         ../conda/bin/python python/train/train_singlge_cws4.py  --word2vec_path vec.txt --train_data_path /data0/xijun1/tfqieci/train.txt  --test_data_path test.txt --max_sentence_len 200  --learning_rate 0.001
 
     demo演示：
            ../conda/bin/python  python/web/index.py 
