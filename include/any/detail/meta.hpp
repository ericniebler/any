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
#include "typeinfo.hpp"

#include <algorithm>
#include <array>
#include <span>

ANY_DIAG_PUSH()
ANY_DIAG_SUPPRESS_MSVC(4141) // 'inline' used more than once

namespace any
{
template <class...>
struct _undef;

template <class Fn, class... Args>
using _mcall = Fn::template call<Args...>;

//////////////////////////////////////////////////////////////////////////////////////////
// _if_t
template <bool>
struct _if_
{
  template <class Then, class Else>
  using call = Then;
};

template <>
struct _if_<false>
{
  template <class Then, class Else>
  using call = Else;
};

template <bool Condition, class Then, class Else>
using _if_t = _mcall<_if_<Condition>, Then, Else>;

//////////////////////////////////////////////////////////////////////////////////////////
// _copy_cvref_t
template <class T, int = 0>
extern _undef<T> _copy_cvref_fn;

#define ANY_COPY_CVREF(NAME, QUAL)                                                                 \
  struct NAME                                                                                      \
  {                                                                                                \
    template <class T>                                                                             \
    using call = T QUAL;                                                                           \
  };                                                                                               \
  template <class T>                                                                               \
  extern NAME _copy_cvref_fn<T QUAL, 0>

ANY_COPY_CVREF(_cp, );
ANY_COPY_CVREF(_cpl, &);
ANY_COPY_CVREF(_cpr, &&);
ANY_COPY_CVREF(_cpc, const);
ANY_COPY_CVREF(_cpcl, const &);
ANY_COPY_CVREF(_cpcr, const &&);

#undef ANY_COPY_CVREF

template <class From, class To>
using _copy_cvref_t = _mcall<decltype(_copy_cvref_fn<From>), To>;

//////////////////////////////////////////////////////////////////////////////////////////
// _mquote
template <template <class...> class Fn>
struct _mquote
{
  template <class... Args>
  using call = Fn<Args...>;
};

//////////////////////////////////////////////////////////////////////////////////////////
// _mvalue
template <auto Value>
struct _mvalue
{
  using value_type            = decltype(Value);
  static constexpr auto value = Value;

  constexpr operator value_type() const noexcept
  {
    return Value;
  }

  template <class Type>
    requires std::convertible_to<value_type const &, std::span<Type>>
  constexpr operator std::span<Type>() const noexcept
  {
    return value;
  }

  static constexpr auto begin() noexcept
    requires std::ranges::range<value_type>
  {
    return std::ranges::begin(value);
  }

  static constexpr auto end() noexcept
    requires std::ranges::range<value_type>
  {
    return std::ranges::end(value);
  }

  template <auto Other>
  constexpr auto operator+(_mvalue<Other>) noexcept -> _mvalue<Value + Other>
  {
    return {};
  }
};

//////////////////////////////////////////////////////////////////////////////////////////
// _mlist
template <class... Types>
struct _mlist;

//////////////////////////////////////////////////////////////////////////////////////////
// _mindirect
template <class Fn>
struct _mindirect;

namespace detail
{
template <class...>
concept _mtrue = true;

template <bool>
struct _mindirect_
{
  template <class Fn, class... Args>
  using call = _mcall<Fn, Args...>;
};
} // namespace detail

template <class Fn>
struct _mindirect
{
  template <class... Args>
  using call = _mcall<detail::_mindirect_<detail::_mtrue<Args...>>, Fn, Args...>;
};

//////////////////////////////////////////////////////////////////////////////////////////
// _mquote_indirect
template <template <class...> class Fn>
using _mquote_indirect = _mindirect<_mquote<Fn>>;

//////////////////////////////////////////////////////////////////////////////////////////
// _mfor:: [a] -> (a -> b) -> [b]
template <class List>
struct _mfor;

template <template <class...> class List, class... Ts>
struct _mfor<List<Ts...>>
{
  template <class Fn, class... Us>
  using call = _mcall<Fn, Us..., Ts...>;
};

template <class Return, class... Args>
struct _mfor<Return(Args...)>
{
  template <class Fn, class... Us>
  using call = _mcall<Fn, Us..., Return, Args...>;
};

namespace detail
{
template <class Indices>
struct _mfor_range;

template <size_t... Is>
struct _mfor_range<std::index_sequence<Is...>>
{
  template <class Fn, class Range, class... Us>
  using call = _mcall<Fn, Us..., any::typeof_t<Range::value[Is]>...>;
};
} // namespace detail

template <auto Range>
  requires std::ranges::range<decltype(Range)>
        && std::same_as<std::ranges::range_value_t<decltype(Range)>, any::type_index>
struct _mfor<_mvalue<Range>>
{
  using _indices_t = std::make_index_sequence<Range.size()>;

  template <class Fn, class... Us>
  using call = _mcall<detail::_mfor_range<_indices_t>, Fn, _mvalue<Range>, Us...>;
};

//////////////////////////////////////////////////////////////////////////////////////////
// _mapply
template <class Fn, class List, class... Us>
using _mapply = _mcall<_mfor<List>, Fn, Us...>;

//////////////////////////////////////////////////////////////////////////////////////////
// _muncurry
template <class Fn>
struct _muncurry
{
  template <class List, class... Us>
  using call = _mapply<Fn, List, Us...>;
};

//////////////////////////////////////////////////////////////////////////////////////////
// _mcount
template <class... Ts>
using _mcount = _mvalue<sizeof...(Ts)>;

//////////////////////////////////////////////////////////////////////////////////////////
// _msize
struct _msize
{
  template <class List>
  using call = _mapply<_mquote<_mcount>, List>;
};

//////////////////////////////////////////////////////////////////////////////////////////
// Type set utilities (_mset)
template <size_t MaxSize>
struct _mset;

template <size_t MaxSize>
constexpr auto _mset_unique(_mset<MaxSize> set) noexcept -> _mset<MaxSize>
{
  std::ranges::sort(set);
  set.size_ -= std::ranges::unique(set).size();
  return set;
}

template <size_t MaxSize>
struct _mset
{
  constexpr auto begin(this auto &self) noexcept
  {
    return self.types_.begin();
  }
  constexpr auto end(this auto &self) noexcept
  {
    return self.types_.begin() + self.size_;
  }
  constexpr auto size() const noexcept
  {
    return size_;
  }
  constexpr any::type_index operator[](size_t i) const noexcept
  {
    return types_[i];
  }
  template <size_t OtherSize>
  constexpr auto operator+(_mset<OtherSize> const &other) const noexcept
      -> _mset<MaxSize + OtherSize>
  {
    _mset<MaxSize + OtherSize> result{};
    std::ranges::copy(types_, result.types_.begin());
    std::ranges::copy(other.types_, result.types_.begin() + size());
    result.size_ = size() + other.size();
    return _mset_unique(result);
  }

  std::array<any::type_index, MaxSize> types_;
  size_t size_ = MaxSize;
};

template <class Type, size_t MaxSize>
constexpr bool _mset_contains(_mset<MaxSize> const &set) noexcept
{
  return std::ranges::contains(set, any::type_index_of<Type>);
}

template <class... Ts>
using _mmake_set = _mvalue<any::_mset_unique(
    _mset{std::array<any::type_index, sizeof...(Ts)>{any::type_index_of<Ts>...}})>;

} // namespace any

ANY_DIAG_POP()
