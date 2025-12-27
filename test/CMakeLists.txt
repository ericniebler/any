add_executable(any_test any_test.cpp)
target_link_libraries(any_test PRIVATE any Catch2::Catch2WithMain)
