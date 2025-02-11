cmake_minimum_required(VERSION 3.18)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include_directories(_inc)
aux_source_directory(_src SOURCES)
aux_source_directory(_src TEST_SOURCES)
list(FILTER TEST_SOURCES EXCLUDE REGEX ".*main\\.cpp")
aux_source_directory(_test TEST_SOURCES)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

set(EXE_NAME "out.exe")
add_executable(${EXE_NAME} ${SOURCES})
target_compile_definitions(${EXE_NAME} PRIVATE LINUX)
set_target_properties(${EXE_NAME} PROPERTIES COMPILE_FLAGS "-m64" LINK_FLAGS "-m64")

# Włączenie testów
enable_testing()

# Pobieranie jako submoduł
add_subdirectory(external/googletest)

# Dodanie testów
set(TEST_EXE_NAME "test.texe")  # Nazwa binarki testowej
add_executable(${TEST_EXE_NAME} ${TEST_SOURCES})
target_link_libraries(${TEST_EXE_NAME} PRIVATE gtest_main)

# Rejestracja testów
include(GoogleTest)
gtest_discover_tests(${TEST_EXE_NAME})
