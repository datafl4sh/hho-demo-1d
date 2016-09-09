cmake_minimum_required (VERSION 2.8.11)
project (hho-demo-1d)

set(CMAKE_CXX_FLAGS_DEBUG "-std=c++14 -g -Wall -Wshadow")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wall -Wshadow -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASEASSERT "-std=c++14 -Wall")

find_package(Armadillo REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif(NOT CMAKE_BUILD_TYPE)

add_executable(hho-demo-1d hho-demo-1d.cpp)
target_link_libraries(hho-demo-1d armadillo boost_iostreams boost_system)

install(TARGETS hho-demo-1d RUNTIME DESTINATION bin)
