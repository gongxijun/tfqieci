include_directories(${PROJECT_SOURCE_DIR}/include)

set(BASE_LIBS gflags glog protobuf)
#create bin word2vec
set(WORD2VEC_LIBS pthread)
add_executable(word2vec_bin word2vec.cc)
target_link_libraries(word2vec_bin ${WORD2VEC_LIBS})

#create
add_executable(distance  distance.cc)


#target_link_libraries(proto ${PROTOBUF_LIBRARIES})