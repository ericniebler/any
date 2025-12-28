CPMAddPackage(NAME Catch2
              VERSION 3.11.0
              GIT_REPOSITORY https://github.com/catchorg/Catch2.git
              GIT_TAG v3.11.0
              OPTIONS "CATCH_BUILD_TESTS OFF"
                      "CATCH_INSTALL_DOCS OFF"
                      "CATCH_INSTALL_HELPERS OFF"
                      "CATCH_ENABLE_CXX20 ON"
              QUIET)

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)

include(CTest)
include(Catch)

add_executable(any_test any_test.cpp)
target_link_libraries(any_test PRIVATE any Catch2::Catch2WithMain)
catch_discover_tests(any_test)
