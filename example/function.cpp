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

#if ANY_COMPILER_CLANG
#  pragma clang diagnostic ignored "-Winfinite-recursion"
#endif

template <class Signature>
struct _ifunction;

template <class Return, class... Args>
struct _ifunction<Return(Args...)>
{
  // "abstract" interface for callable types
  template <class Base>
  struct _interface : any::interface<_interface, Base, any::extends<any::icopyable>>
  {
    using _interface::interface::interface;

    constexpr virtual Return operator()(Args... args) const
    {
      return any::value(*this)(std::forward<Args>(args)...);
    }
  };
};

template <class Signature>
using function = any::any<_ifunction<Signature>::template _interface>;

// A concrete type that models the interface:
struct myfun
{
  void operator()() const
  {
    std::printf("myfun::operator() called\n");
  }
};

int main()
{
  function<void()> fun = myfun{};
  fun();
}
