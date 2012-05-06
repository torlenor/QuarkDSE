# The name of our project is "HELLO". CMakeLists files in this project can
# refer to the root source directory of the project as ${HELLO_SOURCE_DIR} and
# to the root binary directory of the project as ${HELLO_BINARY_DIR}.
cmake_minimum_required (VERSION 2.8)
project (CMELP2)

# Recurse into the "Hello" and "Demo" subdirectories. This does not actually
# cause another cmake executable to run. The same process will walk through
# the project's entire directory structure.

# Create a library called "Hello" which includes the source file "hello.cxx".
# The extension is already found. Any number of sources could be listed here.
# add_library (neib neib.cpp)
# add_library (chargeconfig chargeconfig.cpp

include_directories (${CMELP2_SOURCE_DIR})

# Make sure the linker can find the Hello library once it is built.
link_directories (${CMELP2_BINARY_DIR})

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-O3")        ## Optimize
# set(CMAKE_EXE_LINKER_FLAGS "-s")  ## Strip binary

# Add executable called "helloDemo" that is built from the source files
# "demo.cxx" and "demo_b.cxx". The extensions are automatically found.
# add_executable (calcalljack calcalljack.cpp calcint.cpp calcjack.cpp)
#add_executable (bin/tests tests.cpp)
add_executable (bin/main_cpu main_cpu.cpp)

# Link the executable to the Hello library.
# target_link_libraries (bin/main_cpu gsl blas) 
