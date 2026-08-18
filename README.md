# any

[![CI](https://github.com/ericniebler/any/actions/workflows/ci.yaml/badge.svg)](https://github.com/ericniebler/any/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)

A single-header, C++23 library for building type-erasing wrappers.

`any` lets you define your own type-erased interfaces — the same way
`std::function` erases callables and `std::any` erases everything — without
writing the boilerplate vtable/model/concept machinery by hand. Erased types
don't need to inherit from anything or know about the interface at all.

## Table of contents

- [Example](#example)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [More examples](#more-examples)
- [Building and testing](#building-and-testing)
- [Code coverage](#code-coverage)
- [Contributing](#contributing)
- [License](#license)

## Example

```c++
#include "any/any.hpp"

#include <cassert>
#include <cstdio>

// "abstract" interfaces:
template <class Model>
struct idrawable : any::interface<idrawable, Model>
{
  using idrawable::interface::interface;

  constexpr virtual void draw() const
  {
    any::value(*this).draw();
  }
};

namespace my
{
// A concrete type that models the interface but
// that does not inherit from it:
struct drawable
{
  void draw() const
  {
    std::printf("my::drawable::draw()\n");
  }
};
} // namespace my

int main()
{
  any::any<idrawable> widget = my::drawable{};

  // prints "my::drawable::draw()":
  widget.draw();
}
```

## Features

- **Header-only.** Drop `include/any` into your project or pull it in with
  CPM/FetchContent — no library to build or link.
- **Non-intrusive.** Any type that structurally satisfies an interface can be
  erased into it, without inheriting from the interface or otherwise
  depending on the `any` library.
- **Composable interfaces.** Interfaces can extend other interfaces (see
  `any::extends`), so capabilities like copyability or comparability can be
  mixed in independently of any interfaces you define.
- **`constexpr`-friendly.** Interfaces are written as ordinary virtual
  functions and work at compile time where the rest of your code does.

## Requirements

- A C++23 compiler. CI builds and tests against clang 22 and gcc 14 on
  Linux, and MSVC on Windows.
- CMake 3.10+ if you're using the provided build (not required to just
  copy the headers into your own project).

## Installation

### CMake (CPM or FetchContent)

```cmake
CPMAddPackage(
  NAME any
  GIT_REPOSITORY https://github.com/ericniebler/any.git
  GIT_TAG main)

target_link_libraries(your_target PRIVATE any)
```

### Copy the headers

`any` is header-only with no dependencies, so you can also just copy
`include/any` into your project's include path.

## More examples

The [`example`](example) directory has more complete demonstrations:

- [`basic.cpp`](example/basic.cpp) — the example above.
- [`function.cpp`](example/function.cpp) — erasing callables into a
  `std::function`-like interface.
- [`comparable.cpp`](example/comparable.cpp) — erasing binary operators such
  as equality comparison.
- [`queries.cpp`](example/queries.cpp) — querying whether an erased value
  models an optional, extended interface.

## Building and testing

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

## Code coverage

Coverage is measured with clang's source-based coverage (`llvm-cov`) and
reported for every push to `main`:

**[View the latest coverage report](https://ericniebler.github.io/any/coverage/)**

To generate it locally with clang:

```sh
cmake -S . -B build -DBUILD_TESTING=ON -DANY_ENABLE_COVERAGE=ON \
  -DCMAKE_CXX_COMPILER=clang++
cmake --build build --target coverage
# open build/coverage/html/index.html
```

## Contributing

Issues and pull requests are welcome. Please make sure `ctest` passes and
new code is covered by tests before opening a PR.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
