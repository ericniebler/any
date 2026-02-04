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

#include "any/any.hpp"

#include <cassert>
#include <concepts>
#include <stop_token>
#include <utility>

// This header defines some queries and some type-erasing wrappers for holding the
// values of type-erased queries.

ANY_DIAG_PUSH()
ANY_DIAG_SUPPRESS_CLANG("-Winfinite-recursion")

namespace detail
{
template <class T, class U>
concept _not_same_as = !std::same_as<T, U>;

template <class Queryable, class Query>
concept _queryable_with = requires(Queryable const &q) {
  { q.query(Query{}) } -> _not_same_as<void>;
};

template <class Queryable, class Query>
concept _nothrow_queryable_with = requires(Queryable const &q) {
  { q.query(Query{}) } noexcept -> _not_same_as<void>;
};

template <class Queryable, class Query>
using _query_result_t =
    decltype(std::declval<Queryable const &>().query(std::declval<Query const &>()));

template <class Fn, class Default>
struct _with_default : Fn
{
  using Fn::operator();
  constexpr Default operator()(any::_ignore) const noexcept
  {
    return default_;
  }
  Default default_;
};

template <class Query>
struct _query_value
{
  using type = any::any<any::icopyable>;
};

template <class Query>
  requires requires { typename Query::value_type; }
struct _query_value<Query>
{
  using type = Query::value_type;
};
} // namespace detail

//////////////////////////////////////////////////////////////////////////////////////////
// The (possibly type-erased) value type to be returned by type-erasing Query
template <class Query>
using query_value_t = detail::_query_value<Query>::type;

//////////////////////////////////////////////////////////////////////////////////////////
// Whether the query should always be noexcept
template <class Query>
inline constexpr bool is_always_nothrow_query = false;

template <class Query>
  requires requires { Query::always_nothrow ? true : false; }
inline constexpr bool is_always_nothrow_query<Query> = Query::always_nothrow;

//////////////////////////////////////////////////////////////////////////////////////////
// Representation of a set of queries. Denotes a constant wrapper around a std::array of
// type_index values. Addition performs set union.
template <class... Queries>
using query_set = any::_mmake_set<Queries...>;

//////////////////////////////////////////////////////////////////////////////////////////
// imalloc
template <class Model>
struct imalloc : any::interface<imalloc, Model, any::extends<any::icopyable>>
{
  using imalloc::interface::interface;

  [[nodiscard]]
  constexpr virtual void *allocate(size_t bytes)
  {
    return any::value(*this).allocate(bytes);
  }

  constexpr virtual void deallocate(void *ptr, size_t bytes) noexcept
  {
    any::value(*this).deallocate(static_cast<std::byte *>(ptr), bytes);
  }
};

//////////////////////////////////////////////////////////////////////////////////////////
// any_allocator
template <class Type>
struct any_allocator
{
  using value_type = Type;

  any_allocator()  = default;

  template <detail::_not_same_as<any_allocator> Alloc>
  constexpr any_allocator(Alloc alloc)
    : malloc_(_malloc_t<Alloc>(std::move(alloc)))
  {
    using other_value_type = std::allocator_traits<Alloc>::value_type;
    static_assert(std::same_as<other_value_type, Type>);
  }

  template <detail::_not_same_as<Type> Other>
  constexpr any_allocator(any_allocator<Other> other) noexcept
    : malloc_(std::move(other.malloc_))
  {
  }

  [[nodiscard]]
  constexpr Type *allocate(size_t count)
  {
    return static_cast<Type *>(malloc_.allocate(count * sizeof(Type)));
  }

  constexpr void deallocate(Type *ptr, size_t count) noexcept
  {
    malloc_.deallocate(ptr, count * sizeof(Type));
  }

  template <class Alloc, class... Args>
  constexpr void emplace(Args &&...args)
  {
    using other_value_type = std::allocator_traits<Alloc>::value_type;
    static_assert(std::same_as<other_value_type, Type>);
    _malloc_t<Alloc> malloc(Alloc(std::forward<Args>(args)...));
    malloc_.emplace(std::move(malloc));
  }

  template <int = 0, class Alloc>
  constexpr void emplace(Alloc alloc)
  {
    emplace<Alloc>(std::move(alloc));
  }

private:
  template <class Alloc>
  using _malloc_t = std::allocator_traits<Alloc>::template rebind_alloc<std::byte>;

  template <class>
  friend struct any_allocator;

  friend any::access;

  [[nodiscard]]
  constexpr bool _empty_() const noexcept
  {
    return any::empty(malloc_);
  }

  any::any<imalloc> malloc_;
};

//////////////////////////////////////////////////////////////////////////////////////////
// icallback - a type-erased nullary callable
template <class Model>
struct icallback : any::interface<icallback, Model, any::extends<any::icopyable>>
{
  using icallback::interface::interface;

  constexpr virtual void operator()() noexcept
  {
    (any::value(*this))();
  }
};

//////////////////////////////////////////////////////////////////////////////////////////
// istop_callback
template <class Model>
struct istop_callback : any::interface<istop_callback, Model>
{
  using istop_callback::interface::interface;
};

namespace detail
{
struct _register_callback_fn;
} // namespace detail

//////////////////////////////////////////////////////////////////////////////////////////
// istop_token
template <class Model>
struct istop_token
  : any::interface<istop_token, Model, any::extends<any::icopyable, any::iequality_comparable>>
{
  using istop_token::interface::interface;

  [[nodiscard]]
  constexpr virtual bool stop_requested() const noexcept
  {
    return any::value(*this).stop_requested();
  }

  [[nodiscard]]
  constexpr virtual bool stop_possible() const noexcept
  {
    return any::value(*this).stop_possible();
  }

private:
  template <class>
  friend struct istop_token;

  friend struct detail::_register_callback_fn;

  [[nodiscard]]
  constexpr virtual auto _register_callback(any::any<icallback> callback)
      -> any::any<istop_callback>
  {
    return any::value(*this)._register_callback(std::move(callback));
  }
};

namespace detail
{
//////////////////////////////////////////////////////////////////////////////////////////
// function object for registering a stop callback granted friendship to istop_token
struct _register_callback_fn
{
  [[nodiscard]]
  constexpr auto operator()() const -> any::any<istop_callback>
  {
    return token._register_callback(std::move(callback));
  }

  any::any<istop_token> &token;
  any::any<icallback> &callback;
};

//////////////////////////////////////////////////////////////////////////////////////////
// _stop_callback_for to work around the fact that std::stop_callback does not have a
// nested ::callback_type alias template.
template <class Token>
struct _stop_callback_for
{
  template <class Callback>
  using call = Token::template callback_type<Callback>;
};

template <>
struct _stop_callback_for<std::stop_token>
{
  template <class Callback>
  using call = std::stop_callback<Callback>;
};
} // namespace detail

//////////////////////////////////////////////////////////////////////////////////////////
// never_stop_token
struct never_stop_token
{
  struct _callback
  {
    constexpr explicit _callback(never_stop_token, any::_ignore) noexcept
    {
    }
  };

  template <class>
  using callback_type = _callback;

  [[nodiscard]]
  static constexpr bool stop_requested() noexcept
  {
    return false;
  }

  [[nodiscard]]
  static constexpr bool stop_possible() noexcept
  {
    return false;
  }

  bool operator==(never_stop_token const &) const noexcept = default;
};

//////////////////////////////////////////////////////////////////////////////////////////
// any_stop_token
struct any_stop_token
{
private:
  struct _callback;

public:
  template <class Callback>
  using callback_type                 = _callback;

  constexpr any_stop_token() noexcept = default;

  template <detail::_not_same_as<any_stop_token> Token>
  constexpr any_stop_token(Token token)
    : token_(_token_wrapper(std::move(token)))
  {
  }

  [[nodiscard]]
  constexpr bool stop_requested() const noexcept
  {
    return token_.stop_requested();
  }

  [[nodiscard]]
  constexpr bool stop_possible() const noexcept
  {
    return token_.stop_possible();
  }

  template <int = 0, class Token>
  constexpr auto emplace(Token token) -> Token &
  {
    return token_.emplace(_token_wrapper(std::move(token)));
  }

  template <class Token, class... Args>
  constexpr auto emplace(Args &&...args) -> Token &
  {
    return token_.template emplace<_token_wrapper<Token>>(std::forward<Args>(args)...);
  }

private:
  friend struct any::access; // so that any::empty can call _empty_()

  ////////////////////////////////////////////////////////////////////////////////////////
  // adds to Token a factory function for registering a stop callback
  template <class Token>
  struct _token_wrapper : Token
  {
    constexpr explicit _token_wrapper(Token token)
      : Token(std::move(token))
    {
    }

    template <class... Args>
    constexpr explicit _token_wrapper(Args &&...args)
      : Token(std::forward<Args>(args)...)
    {
    }

    [[nodiscard]]
    constexpr auto _register_callback(any::any<icallback> callback) -> any::any<istop_callback>
    {
      Token &token     = *this;
      using callback_t = any::_mcall<detail::_stop_callback_for<Token>, any::any<icallback>>;
      return any::any<istop_callback>(std::in_place_type<callback_t>, token, std::move(callback));
    }
  };

  struct _callback
  {
    explicit _callback(any_stop_token &token, any::any<icallback> callback)
    {
      callback_.emplace(any::_emplace_from{detail::_register_callback_fn{token.token_, callback}});
    }

  private:
    any::any<istop_callback> callback_{};
  };

  constexpr bool _empty_() const noexcept
  {
    return any::empty(token_);
  }

  any::any<istop_token> token_;
};

//////////////////////////////////////////////////////////////////////////////////////////
// get_queries
struct get_queries_t
{
  using value_type                     = std::span<any::type_index const>;
  static constexpr bool always_nothrow = true;

  template <detail::_queryable_with<get_queries_t> Queryable>
  static constexpr auto operator()(Queryable const &q) noexcept
      -> detail::_query_result_t<Queryable, get_queries_t>
  {
    static_assert(noexcept(q.query(get_queries_t{})),
                  "Queryable::query must be noexcept for get_queries queries");
    return q.query(get_queries_t{});
  }
};

inline constexpr auto get_queries = get_queries_t{};

//////////////////////////////////////////////////////////////////////////////////////////
// get_allocator
struct get_allocator_t
{
  // Used as the query result when the get_allocator query is type-erased
  using value_type                     = any_allocator<std::byte>;
  static constexpr bool always_nothrow = true;

  template <detail::_queryable_with<get_allocator_t> Queryable>
  static constexpr auto operator()(Queryable const &q) noexcept
      -> detail::_query_result_t<Queryable, get_allocator_t>
  {
    static_assert(noexcept(q.query(get_allocator_t{})),
                  "Queryable::query must be noexcept for get_allocator queries");
    return q.query(get_allocator_t{});
  }
};

inline constexpr get_allocator_t get_allocator{};

//////////////////////////////////////////////////////////////////////////////////////////
// get_scheduler
struct get_scheduler_t
{
  static constexpr bool always_nothrow = true;

  template <detail::_queryable_with<get_scheduler_t> Queryable>
  static constexpr auto operator()(Queryable const &q) noexcept
      -> detail::_query_result_t<Queryable, get_scheduler_t>
  {
    static_assert(noexcept(q.query(get_scheduler_t{})),
                  "Queryable::query must be noexcept for get_scheduler queries");
    return q.query(get_scheduler_t{});
  }
};

inline constexpr auto get_scheduler = get_scheduler_t{};

//////////////////////////////////////////////////////////////////////////////////////////
// get_stop_token
struct get_stop_token_t
{
  using value_type = any_stop_token;

  template <detail::_queryable_with<get_stop_token_t> Queryable>
  static constexpr auto operator()(Queryable const &q) noexcept(
      detail::_nothrow_queryable_with<Queryable, get_stop_token_t>)
      -> detail::_query_result_t<Queryable, get_stop_token_t>
  {
    return q.query(get_stop_token_t{});
  }

  static constexpr auto operator()(any::_ignore) noexcept -> never_stop_token
  {
    return never_stop_token{};
  }
};

inline constexpr auto get_stop_token = get_stop_token_t{};

ANY_DIAG_POP()
