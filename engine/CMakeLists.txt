cmake_minimum_required(VERSION 3.20)
project(braid_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The engine is a submodule, pinned to a commit. Building it from here rather
# than expecting a preinstalled copy means the checkpoint format, the model
# geometry and the kernels always come from the same revision as the code that
# reads them.
#
# ENGINE_CUDA defaults to ON here where it defaults to OFF upstream: a worker
# built without the GPU path still works, but it is not the thing this project
# is about, so making it opt-out rather than opt-in keeps a CPU-only build from
# being produced by accident and then measured.
option(BRAID_CUDA "Build the worker against the engine's CUDA backend" ON)
set(ENGINE_CUDA ${BRAID_CUDA} CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cpp-ai-engine
                 ${CMAKE_BINARY_DIR}/cpp-ai-engine EXCLUDE_FROM_ALL)

add_executable(braid_worker worker.cpp)
target_link_libraries(braid_worker PRIVATE engine)
target_include_directories(braid_worker PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(braid_train train.cpp)
target_link_libraries(braid_train PRIVATE engine)
target_include_directories(braid_train PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
