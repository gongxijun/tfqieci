/*
 * Copyright 2016- 2018 Koth. All Rights Reserved.
 * =====================================================================================
 * Filename:  gen_seg_eval.cc
 * Author:  Koth
 * Create Time: 2016-11-29 09:26:39
 * Description:
 *
 */
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>

#include "base/base.h"
#include "utils/basic_string_util.h"


#include "cc/tf_seg_model.h"  //NOLINT
#include "cc/sentence_breaker.h"  // NOLINT
#include "tensorflow/core/platform/init_main.h"

DEFINE_string(test_file, "", "the test file");
DEFINE_string(model_path, "", "the model path");
DEFINE_string(vocab_path, "", "vocab path");

DEFINE_int32(max_setence_len, 80, "max sentence len");

const int BATCH_SIZE = 2000;
int load_test_file(const std::string& path,
                   std::vector<std::string>* pstrs) {
  FILE *fp = fopen(path.c_str(), "r");
  if (fp == NULL) {
    VLOG(0) << "open file error:" << path;
    return 0;
  }
  char line[4096] = {0};
  int tn = 0;
  while (fgets(line, sizeof(line) - 1, fp)) {
    int nn = strlen(line);
    while (nn && (line[nn - 1] == '\n' || line[nn - 1] == '\r')) {
      nn -= 1;
    }
    if (nn <= 0) {
      continue;
    }
    pstrs->push_back(std::string(line, nn));
    tn += 1;
  }
  fclose(fp);
  return tn;
}
int main(int argc, char *argv[]) {
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_vocab_path.empty()) {
    VLOG(0) << "basic bocab path is not set";
    return 1;
  }
  if (FLAGS_model_path.empty()) {
    VLOG(0) << " model path is not set";
    return 1;
  }
  if (FLAGS_test_file.empty()) {
    VLOG(0) << " test_file path is not set";
    return 1;
  }
  FILE* outfp = fopen("out_eval.txt", "w");
  CHECK(outfp != nullptr) << "open file 'out_eval.txt' error";
  kcws::TfSegModel sm;
  CHECK(sm.LoadModel(FLAGS_model_path,
                     FLAGS_vocab_path,
                     FLAGS_max_setence_len))
      << "Load model error";

  std::vector<std::string> teststrs;
  int ns = load_test_file(FLAGS_test_file, &teststrs);
  std::string todo;
  for (int i = 0; i < ns; i++) {
    todo.append(teststrs[i]);
  }
  VLOG(0) << "loaded :" << FLAGS_test_file << " ,got " << ns << " lines";

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < ns; i++) {
    // VLOG(0) << "do line:" << i;
    if (teststrs[i].empty()) {
      VLOG(0) << "empty line , continue";
      continue;
    }
    std::vector<std::string> results;
    CHECK(sm.Segment(teststrs[i], &results)) << "segment error";
    int nr = results.size();
    CHECK_NE(nr, 0);
    fprintf(outfp, "%s", results[0].c_str());
    for (int i = 1; i < nr; i++) {
      fprintf(outfp, " %s", results[i].c_str());
    }
    fprintf(outfp, "\n");
  }
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
                  (std::chrono::steady_clock::now() - start);
  VLOG(0) << "spend " << duration.count() << " milliseconds for file:" << FLAGS_test_file;

  return 0;
}
