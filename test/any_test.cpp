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

#include <cstdio>

#include <concepts>
#include <type_traits>

#include "catch2/catch_all.hpp" // IWYU pragma: keep

template <class Base>
struct ifoo : any::interface<ifoo, Base>
{
  using ifoo::interface::interface;

  constexpr virtual void foo()
  {
    any::value(*this).foo();
  }

  constexpr virtual void cfoo() const
  {
    any::value(*this).cfoo();
  }
};

template <class Base>
struct ibar : any::interface<ibar, Base, any::extends<ifoo, any::icopyable>>
{
  using ibar::interface::interface;

  constexpr virtual void bar()
  {
    any::value(*this).bar();
  }
};

template <class Base>
struct ibaz : any::interface<ibaz, Base, any::extends<ibar>, 5 * sizeof(void *)>
{
  using ibaz::interface::interface;

  constexpr ~ibaz() = default;

  constexpr virtual void baz()
  {
    any::value(*this).baz();
  }
};

struct foobar
{
  constexpr void foo()
  {
    if (!std::is_constant_evaluated())
      std::printf("foo override, value = %d\n", value);
  }

  constexpr void cfoo() const
  {
    if (!std::is_constant_evaluated())
      std::printf("cfoo override, value = %d\n", value);
  }

  constexpr void bar()
  {
    if (!std::is_constant_evaluated())
      std::printf("bar override, value = %d\n", value);
  }

  constexpr void baz()
  {
    if (!std::is_constant_evaluated())
      std::printf("baz override, value = %d\n", value);
  }

  bool operator==(foobar const &other) const noexcept = default;

  int value                                           = 42;
};

static_assert(std::derived_from<any::iabstract<any::icopyable>, any::iabstract<any::imovable>>);
static_assert(std::derived_from<any::iabstract<ibar>, any::iabstract<ifoo>>);
static_assert(!std::derived_from<any::iabstract<ibar>, any::iabstract<any::icopyable>>);
static_assert(any::extension_of<any::iabstract<ibar>, any::icopyable>);

// Test the Diamond of Death inheritance problem:
template <class Base>
struct IFoo : any::interface<IFoo, Base, any::extends<any::icopyable>>
{
  using IFoo::interface::interface;

  constexpr virtual void foo()
  {
    any::value(*this).foo();
  }
};

template <class Base>
struct IBar : any::interface<IBar, Base, any::extends<any::icopyable>>
{
  using IBar::interface::interface;

  constexpr virtual void bar()
  {
    any::value(*this).bar();
  }
};

template <class Base>
struct IBaz : any::interface<IBaz, Base, any::extends<IFoo, IBar>> // inherits twice
                                                                   // from icopyable
{
  using IBaz::interface::interface;

  constexpr virtual void baz()
  {
    any::value(*this).baz();
  }
};

static_assert(std::derived_from<any::iabstract<IBaz>, any::iabstract<IFoo>>);
static_assert(std::derived_from<any::iabstract<IBaz>, any::iabstract<any::icopyable>>);

void test_deadly_diamond_of_death()
{
  any::any<IBaz> m(foobar{});

  m.foo();
  m.bar();
  m.baz();
}

static_assert(any::iabstract<ifoo>::buffer_size < any::iabstract<ibaz>::buffer_size);

// test constant evaluation works
consteval void test_consteval()
{
  any::any<ibaz> m(foobar{});
  [[maybe_unused]] auto x = any::any_static_cast<foobar>(m);
  x                       = any::any_cast<foobar>(m);
  m.foo();
  [[maybe_unused]] auto n               = m;
  [[maybe_unused]] auto p               = any::caddressof(m);

  any::any<any::iequality_comparable> a = 42;
  if (a != a)
    throw "error";

  any::any_ptr<ibaz> pifoo = any::addressof(m);
  [[maybe_unused]] auto y  = any::any_cast<foobar>(pifoo);
}

TEST_CASE("basic usage", "[any]")
{
  std::printf("%.*s\n", (int)ANY_TYPEID(foobar).name().size(), ANY_TYPEID(foobar).name().data());
  std::printf("sizeof void*: %d\n", (int)sizeof(void *));
  std::printf("sizeof interface: %d\n", (int)sizeof(any::iabstract<ibaz>));

#if ANY_COMPILER_CLANG || ANY_COMPILER_GCC >= 14'03
  test_consteval();
#endif

  any::any<ibaz> m(foobar{});
  REQUIRE(m._in_situ());
  REQUIRE(any::type(m) == ANY_TYPEID(foobar));

  m.foo();
  m.bar();
  m.baz();

  any::any<ifoo> n = std::move(m);
  n.foo();

  auto ptr = any::caddressof(m);
  any::_unconst(*ptr).foo();
  // ptr->foo(); // does not compile because it is a const-correctness violation
  ptr->cfoo();
  auto const ptr2 = any::addressof(m);
  ptr2->foo();
  any::any_ptr<ifoo> pifoo      = ptr2;
  m                             = *ptr; // assignment from type-erased references is supported

  any::any<any::isemiregular> a = 42;
  any::any<any::isemiregular> b = 42;
  any::any<any::isemiregular> c = 43;
  REQUIRE(a == b);
  REQUIRE(!(a != b));
  REQUIRE(!(a == c));
  REQUIRE(a != c);

  any::reset(b);
  REQUIRE(!(a == b));
  REQUIRE(a != b);
  REQUIRE(!(b == a));
  REQUIRE(b != a);

  any::any<any::iequality_comparable> x = a;
  REQUIRE(x == x);
  REQUIRE(x == a);
  REQUIRE(a == x);
  a = 43;
  REQUIRE(x != a);
  REQUIRE(a != x);

  any::reset(a);
  REQUIRE(b == a);

  auto z                        = any::caddressof(c);
  [[maybe_unused]] int const *p = &any::any_cast<int>(c);
  [[maybe_unused]] int const *q = any::any_cast<int>(z);

  REQUIRE(any::any_cast<int>(z) == &any::any_cast<int>(c));

  auto y = any::addressof(c);
  int *r = any::any_cast<int>(std::move(y));
  REQUIRE(r == &any::any_cast<int>(c));

  z = y; // assign non-const ptr to const ptr
  z = &*y;

  REQUIRE(y == z);

  test_deadly_diamond_of_death();
}
