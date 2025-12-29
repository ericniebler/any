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

#pragma once

#include "config.hpp"

#include <cstdarg>
#include <cstdio>

#include <exception>
#include <new>
#include <type_traits>
#include <utility> // IWYU pragma: keep for std::unreachable

ANY_DIAG_PUSH
ANY_DIAG_SUPPRESS_MSVC(4141) // 'inline' used more than once

namespace any
{
//////////////////////////////////////////////////////////////////////////////////////////
// start_lifetime_as
#if __cpp_lib_start_lifetime_as
using std::start_lifetime_as;
#else
template <class T>
[[ANY_ALWAYS_INLINE, nodiscard]]
inline T *start_lifetime_as(void *p) noexcept
{
  return std::launder(static_cast<T *>(p));
}

template <class T>
[[ANY_ALWAYS_INLINE, nodiscard]]
inline T const *start_lifetime_as(void const *p) noexcept
{
  return std::launder(static_cast<T const *>(p));
}
#endif

#if __cpp_lib_unreachable
using std::unreachable;
#else
[[noreturn]] inline void unreachable()
{
  // Uses compiler specific extensions if possible.
  // Even if no extension is used, undefined behavior is still raised by
  // an empty function body and the noreturn attribute.
#  if defined(_MSC_VER) && !defined(__clang__) // MSVC
  __assume(false);
#  else                                        // GCC, Clang
  __builtin_unreachable();
#  endif
}
#endif

template <class Return = void>
[[noreturn]]
inline constexpr Return _die(char const *msg, ...) noexcept
{
  if ANY_CONSTEVAL
  {
    ::any::unreachable();
  }
  else
  {
    va_list args;
    va_start(args, msg);
    std::vfprintf(stderr, msg, args);
    std::fflush(stderr);
    va_end(args);
    std::terminate();
  }
}

template <class T, class U>
concept _decays_to = std::same_as<std::decay_t<T>, U>;

template <class T>
concept _decayed = _decays_to<T, T>;

template <class...>
struct _undef;

template <class Fn, class... Args>
using _mcall = Fn::template call<Args...>;

template <bool>
struct _if_
{
  template <class Then, class...>
  using call = Then;
};

template <>
struct _if_<false>
{
  template <class, class Else>
  using call = Else;
};

template <bool Condition, class Then = void, class... Else>
using _if_t = _mcall<_if_<Condition>, Then, Else...>;

//////////////////////////////////////////////////////////////////////////////////////////
// _copy_cvref_t
#define ANY_COPY_CVREF(NAME, QUAL)                                                                 \
  struct NAME                                                                                      \
  {                                                                                                \
    template <class T>                                                                             \
    using call = T QUAL;                                                                           \
  };                                                                                               \
  template <class T>                                                                               \
  extern NAME _copy_cvref_fn<T QUAL, 0>

template <class T, int = 0>
extern _undef<T> _copy_cvref_fn;

ANY_COPY_CVREF(_cp, );
ANY_COPY_CVREF(_cpl, &);
ANY_COPY_CVREF(_cpr, &&);
ANY_COPY_CVREF(_cpc, const);
ANY_COPY_CVREF(_cpcl, const &);
ANY_COPY_CVREF(_cpcr, const &&);

template <class From, class To>
using _copy_cvref_t = _mcall<decltype(_copy_cvref_fn<From>), To>;

#undef ANY_COPY_CVREF

//////////////////////////////////////////////////////////////////////////////////////////
// _unconst
template <class T>
[[ANY_ALWAYS_INLINE, nodiscard]]
inline constexpr T &_unconst(T const &t) noexcept
{
  return const_cast<T &>(t);
}

//////////////////////////////////////////////////////////////////////////////////////////
// _const_if
template <bool MakeConst, class T>
using _const_if = _if_t<MakeConst, T const, T>;

//////////////////////////////////////////////////////////////////////////////////////////
// _as_const_if
template <bool MakeConst, class T>
[[ANY_ALWAYS_INLINE, nodiscard]]
inline constexpr auto &_as_const_if(T &t) noexcept
{
  return const_cast<_const_if<MakeConst, T> &>(t);
}

//////////////////////////////////////////////////////////////////////////////////////////
// _move_if
template <bool Move, class T>
[[ANY_ALWAYS_INLINE, nodiscard]]
inline constexpr auto &&_move_if(T &t) noexcept
{
  if constexpr (Move)
    return std::move(t);
  else
    return t;
}

//////////////////////////////////////////////////////////////////////////////////////////
// _polymorphic_downcast
template <class ResultPtr, class CvInterface>
[[nodiscard]]
inline constexpr auto *_polymorphic_downcast(CvInterface *from) noexcept
{
  static_assert(std::is_pointer_v<ResultPtr>);
  using value_type = _const_if<std::is_const_v<CvInterface>, std::remove_pointer_t<ResultPtr>>;
  static_assert(std::derived_from<value_type, CvInterface>,
                "_polymorphic_downcast requires From to be a base class of To");

#if __cpp_rtti
  ANY_ASSERT(dynamic_cast<value_type *>(from) != nullptr);
#endif
  return static_cast<value_type *>(from);
}

} // namespace any

ANY_DIAG_POP
