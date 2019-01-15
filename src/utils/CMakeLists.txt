include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/third_party  ${PROJECT_SOURCE_DIR}/python/python_include)
set(W2V_LIBS gflags glog protobuf)
set(W2V_SRC word2vec_vob.cc ${PROJECT_SOURCE_DIR}/include/utils/basic_string_util.h)
#create bin word2vec
#-fPIC 的参数不能丢，PIC(Position Independent Code)表示生成代码与位置无关，这样才能达到动态链接的目的。
set(CMAKE_CXX_FLAGS "-std=c++11 -shared -fPIC -O3 -g ")#-fopenmp

#create
pybind11_add_module(w2v SHARED py_word2vec_vob.cc ${W2V_SRC})
target_link_libraries(w2v PRIVATE ${W2V_LIBS} ${BASE_LIB})
