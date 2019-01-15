/*
 * Copyright 2016- 2018 Koth. All Rights Reserved.
 * =====================================================================================
 * Filename:  seg_backend_api.cc
 * Author:  Koth
 * Create Time: 2016-11-20 20:43:26
 * Description:
 *
 */
#include <string>
#include <thread>
#include <memory>

#include "base/base.h"
#include "utils/jsonxx.h"
#include "utils/basic_string_util.h"
#include "cc/tf_seg_model.h"
#include "cc/pos_tagger.h"
#include "tensorflow/core/platform/init_main.h"
DEFINE_string(model_path, "/home/gxjun/CLionProjects/tfqieci/model/seg_model.pbtxt", "the model path");
DEFINE_string(vocab_path, "/home/gxjun/CLionProjects/tfqieci/model/basic_vocab.txt", "char vocab path");
//DEFINE_string(pos_model_path, "kcws/models/pos_model.pbtxt", "the pos tagging model path");
//DEFINE_string(word_vocab_path, "kcws/models/word_vocab.txt", "word vocab path");
//DEFINE_string(pos_vocab_path, "kcws/models/pos_vocab.txt", "pos vocab path");
DEFINE_int32(max_sentence_len, 80, "max sentence len ");
DEFINE_string(user_dict_path, "", "user dict path");
DEFINE_int32(max_word_num, 50, "max num of word per sentence ");

int main(int argc, char* argv[]) {
  LOG(INFO)<<"test";
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  kcws::TfSegModel model;
  CHECK(model.LoadModel(FLAGS_model_path,
                        FLAGS_vocab_path,
                        FLAGS_max_sentence_len,
                        FLAGS_user_dict_path))
      << "Load model error";
//  if (!FLAGS_pos_model_path.empty()) {
//    kcws::PosTagger* tagger = new kcws::PosTagger;
//    CHECK(tagger->LoadModel(FLAGS_pos_model_path,
//                            FLAGS_word_vocab_path,
//                            FLAGS_vocab_path,
//                            FLAGS_pos_vocab_path,
//                            FLAGS_max_word_num)) << "load pos model error";
//    model.SetPosTagger(tagger);
//  }
  std::string sentence="赵雅淇洒泪道歉 和林丹没有任何经济关系";
  std::vector<std::string> result;
  std::vector<std::string> tags;
  if (model.Segment(sentence, &result, &tags)) {

    jsonxx::Array rarr;
    if (result.size() == tags.size()) {
      int nl = result.size();
      for (int i = 0; i < nl; i++) {
        jsonxx::Object obj;
        obj << "tok" << result[i];
        obj << "pos" << tags[i];
        rarr << obj;
      }
    } else {
      for (std::string str : result) {
        rarr << str;
      }
    }
    LOG(INFO) << "segments" << rarr;
  }

  return 0;
}
