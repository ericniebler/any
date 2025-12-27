#include "any/any.hpp"

#include <cassert>
#include <cstdio>

// "abstract" interfaces:
template <class Model>
struct ibase1 : any::interface<ibase1, Model>
{
  using ibase1::interface::interface;

  constexpr virtual void fn1() const
  {
    any::value(*this).fn1();
  }
};

template <class Model>
struct ibase2 : any::interface<ibase2, Model>
{
  using ibase2::interface::interface;

  constexpr virtual void fn2() const
  {
    any::value(*this).fn2();
  }
};

template <class Model>
struct iderived : any::interface<iderived, Model, any::extends<ibase1, ibase2>>
{
  using iderived::interface::interface;

  constexpr virtual void fn3() const
  {
    any::value(*this).fn3();
  }
};

// A concrete type that models the interface:
struct myfoo
{
  void fn1() const
  {
    std::printf("myfoo::fn1()\n");
  }

  void fn2() const
  {
    std::printf("myfoo::fn2()\n");
  }

  void fn3() const
  {
    std::printf("myfoo::fn3()\n");
  }

  char buffer[128]{};
};

int main()
{
  any::any<iderived> a = myfoo{};
  a.fn1();
  a.fn2();
  a.fn3();

  any::any_ptr<ibase2> p = &a;
  assert(any::data(a) == any::data(*p));
  p->fn2();

  any::iabstract<ibase2> *ptr = &*p;
  std::printf("%s\n", typeid(*ptr).name());
}
