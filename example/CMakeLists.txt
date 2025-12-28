add_executable(basic basic.cpp)
target_link_libraries(basic PRIVATE any)

target_compile_options(
        basic PRIVATE
        $<$<CXX_COMPILER_ID:Clang>:-ftemplate-backtrace-limit=0 -fconstexpr-backtrace-limit=0>
        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -Wpedantic -Werror>)
