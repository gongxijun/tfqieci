include_directories(${PROJECT_SOURCE_DIR}/include)

set(BASE_LIBS gflags glog)
add_library(base SHARED base.cc)
target_link_libraries(base  ${BASE_LIBS})