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
#include "meta.hpp"

#include <cstdarg>
#include <cstdio>

#include <concepts>
#include <exception>
#include <new>
#include <type_traits>
#include <utility> // IWYU pragma: keep for std::unreachable

ANY_DIAG_PUSH
ANY_DIAG_SUPPRESS_MSVC(4141) // 'inline' used more than once

namespace any
{
template <class T, class U>
concept _decays_to = std::same_as<std::decay_t<T>, U>;

template <class T>
concept _decayed = _decays_to<T, T>;

template <class Fn, class... Args>
concept _callable_with =
    requires(Fn &&fn, Args &&...args) { std::forward<Fn>(fn)(std::forward<Args>(args)...); };

template <class Fn, class... Args>
using _call_result_t = decltype(std::declval<Fn>()(std::declval<Args>()...));

struct _ignore
{
  constexpr _ignore(auto &&...) noexcept
  {
  }
};

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
  if consteval
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

//////////////////////////////////////////////////////////////////////////////////////////
// _unconst
template <class T>
[[ANY_ALWAYS_INLINE, nodiscard]]
inline constexpr T &_unconst(T const &t) noexcept
{
  return const_cast<T &>(t);
}

//////////////////////////////////////////////////////////////////////////////////////////
// _as_const_if
template <bool Const, class T>
[[ANY_ALWAYS_INLINE, nodiscard]]
inline constexpr auto &_as_const_if(T &t) noexcept
{
  if constexpr (Const)
    return const_cast<T const &>(t);
  else
    return t;
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
// _emplace_from
template <class Fn>
struct _emplace_from
{
  using type = decltype(std::declval<Fn>()());

  operator type() && noexcept(noexcept(std::declval<Fn>()()))
  {
    return std::move(fn)();
  }

  Fn fn{};
};

//////////////////////////////////////////////////////////////////////////////////////////
// _polymorphic_downcast
template <class ResultPtr, class CvInterface>
[[nodiscard]]
inline constexpr auto *_polymorphic_downcast(CvInterface *from) noexcept
{
  static_assert(std::is_pointer_v<ResultPtr>);
  using value_type = _copy_cvref_t<CvInterface, std::remove_pointer_t<ResultPtr>>;
  static_assert(std::derived_from<value_type, CvInterface>,
                "_polymorphic_downcast requires From to be a base class of To");

#if __cpp_rtti
  if !consteval
  {
    ANY_ASSERT(dynamic_cast<value_type *>(from) != nullptr);
  }
#endif
  return static_cast<value_type *>(from);
}

template <class T>
[[ANY_ALWAYS_INLINE, nodiscard]]
inline constexpr T _decay_copy(T value) noexcept(std::is_nothrow_move_constructible_v<T>)
{
  return value;
}

} // namespace any

ANY_DIAG_POP
