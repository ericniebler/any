function(add_example)
    add_executable(${ARGV0} ${ARGV0}.cpp)
    target_link_libraries(${ARGV0} PRIVATE any)

    target_compile_options(
            ${ARGV0} PRIVATE
            $<$<CXX_COMPILER_ID:Clang>:-ftemplate-backtrace-limit=0 -fconstexpr-backtrace-limit=0 -Winfinite-recursion>
            $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -Wpedantic>)
endfunction()

add_example(basic)
add_example(comparable)
add_example(function)
