/*
 * Copyright (c) 2025 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "any/any.hpp"

#include <cassert>
#include <cstdio>

// This example demonstrates how to type-erase binary operators such as equality
// comparison.

namespace my
{
template <class Base>
struct iequality_comparable : any::interface<iequality_comparable, Base>
{
  using iequality_comparable::interface::interface;

  template <class Other>
  [[nodiscard]]
  constexpr bool operator==(iequality_comparable<Other> const &other) const
  {
    return _equal_to(&other);
  }

private:
  [[nodiscard]]
  constexpr virtual bool _equal_to(any::any_const_ptr<iequality_comparable> other) const
  {
    auto const &type = ::any::type(*this);

    if (type != ::any::type(*other))
      return false;

    if (type == ANY_TYPEID(void))
      return true;

    using value_type = any::value_of_t<iequality_comparable>;
    return any::value(*this) == ::any::any_static_cast<value_type>(*other);
  }
};

template <class Base>
struct isemiregular
  : any::interface<isemiregular, Base, any::extends<any::icopyable, iequality_comparable>>
{
  using isemiregular::interface::interface;
};

using any = any::any<isemiregular>;
} // namespace my

int main()
{
  my::any a = 42;
  my::any b = 42;
  my::any c = 43;
  assert(a == b);
  assert(!(a != b));
  assert(!(a == c));
  assert(a != c);
}
