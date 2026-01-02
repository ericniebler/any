[![CI](https://github.com/ericniebler/any/actions/workflows/ci.yaml/badge.svg)](https://github.com/ericniebler/any/actions/workflows/ci.yaml)

# any

A library for building type-erasing wrappers.

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
  a.draw(); // prints "my::drawable::draw()"
}
```
