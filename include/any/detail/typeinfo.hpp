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

#include <compare>
#include <string_view>

//////////////////////////////////////////////////////////////////////////////////////////
// type_info and ANY_TYPEID

namespace any
{
#define ANY_TYPEID(...) ::any::typeid_of<__VA_ARGS__>

namespace _detail
{
//////////////////////////////////////////////////////////////////////////////////////////
// _pretty_name
template <class>
struct _xyzzy
{
  struct _plugh
  {
  };
};

constexpr char _type_name_prefix[] = "_xyzzy<";
constexpr char _type_name_suffix[] = ">::_plugh";

// Get the type name from the function name by trimming the front and back.
[[nodiscard]]
constexpr std::string_view _find_pretty_name(std::string_view fun_name) noexcept
{
  auto const beg_pos = fun_name.find(_type_name_prefix);
  auto const end_pos = fun_name.rfind(_type_name_suffix);

  auto const start  = beg_pos + sizeof(_type_name_prefix) - 1;
  auto const length = end_pos - start;

  return fun_name.substr(start, length);
}

template <class T>
[[nodiscard]]
constexpr std::string_view _get_pretty_name_helper() noexcept
{
  return _detail::_find_pretty_name(std::string_view{ANY_PRETTY_FUNCTION});
}

template <class T>
[[nodiscard]]
constexpr std::string_view _get_pretty_name() noexcept
{
  return _detail::_get_pretty_name_helper<typename _xyzzy<T>::_plugh>();
}

template <class T>
inline constexpr std::string_view _pretty_name = _detail::_get_pretty_name<T>();

static_assert(_detail::_pretty_name<int> == "int");
} // namespace _detail

//////////////////////////////////////////////////////////////////////////////////////////
// type_info
struct type_info
{
  type_info(type_info &&)            = delete;
  type_info &operator=(type_info &&) = delete;

  constexpr explicit type_info(std::string_view name) noexcept
    : name_(name)
  {
  }

  constexpr std::string_view name() const noexcept
  {
    return name_;
  }

  auto operator==(type_info const &) const noexcept -> bool                  = default;
  auto operator<=>(type_info const &) const noexcept -> std::strong_ordering = default;

private:
  std::string_view name_;
};

template <class T>
inline constexpr type_info typeid_of{_detail::_pretty_name<T>};

template <class T>
inline constexpr type_info const &typeid_of<T const> = typeid_of<T>;

//////////////////////////////////////////////////////////////////////////////////////////
// type_index
struct type_index
{
  constexpr type_index(type_info const &info) noexcept
    : info_(&info)
  {
  }

  constexpr std::string_view name() const noexcept
  {
    return (*info_).name();
  }

  constexpr bool operator==(type_index const &other) const noexcept
  {
    return *info_ == *other.info_;
  }

  constexpr std::strong_ordering operator<=>(type_index const &other) const noexcept
  {
    return *info_ <=> *other.info_;
  }

  type_info const *info_;
};

namespace _detail
{
ANY_DIAG_PUSH
ANY_DIAG_SUPPRESS_GCC("-Wnon-template-friend")
ANY_DIAG_SUPPRESS_NVHPC(probable_guiding_friend)

// The following two classes use the stateful metaprogramming trick to create a spooky
// association between a type_index object and the type it represents.
template <type_index Id>
struct _typeid_c
{
  friend constexpr auto _typeid_lookup(_typeid_c<Id>) noexcept;
};

template <class T>
struct _typeid_of
{
  using type                     = T;
  static constexpr type_index id = type_index(typeid_of<T>);

  friend constexpr auto _typeid_lookup(_typeid_c<id>) noexcept
  {
    return _typeid_of<T>();
  }
};

ANY_DIAG_POP
} // namespace _detail

// For a given type, return a type_index object
template <class T>
inline constexpr type_index type_index_of = _detail::_typeid_of<T>::id;

// For a given type_index object, return the associated type
template <type_index Info>
using typeof_t = typename decltype(_typeid_lookup(_detail::_typeid_c<Info>()))::type;

} // namespace any
