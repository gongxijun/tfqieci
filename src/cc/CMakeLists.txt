include_directories(${PROJECT_SOURCE_DIR}/include)
set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC  -O3 -g -Wno-writable-strings ")#-fopenmp
set(SEG_BASE_LIBS gflags glog protobuf pthread dl)
#create bin word2vec
set(WORD2VEC_LIBS pthread)
set(seg_backend_src tf_seg_model.cc ${PROJECT_SOURCE_DIR}/include/utils/basic_string_util.h ${PROJECT_SOURCE_DIR}/src/utils/basic_vocab.cc
        ${PROJECT_SOURCE_DIR}/src/cc/pos_tagger.cc ${PROJECT_SOURCE_DIR}/src/cc/viterbi_decode.cc
        ${PROJECT_SOURCE_DIR}/src/cc/sentence_breaker.cc  ${PROJECT_SOURCE_DIR}/src/base/base.cc ${PROJECT_SOURCE_DIR}/src/model/tfmodel.cc
        ${PROJECT_SOURCE_DIR}/src/utils/jsonxx.cc)

#create
#add_executable(seg_backend_api.cc ${seg_backend_src})

#target_link_libraries(seg_backend_api ${SEG_BASE_LIBS} ${BASE_LIB} ${_pywrap_tensorflow_internal_LIBRARIES})