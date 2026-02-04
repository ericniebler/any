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
#include <concepts>
#include <stop_token>
#include <utility>

// This header defines some queries and some type-erasing wrappers for holding the
// values of type-erased queries.

ANY_DIAG_PUSH
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
} // namespace detail

//////////////////////////////////////////////////////////////////////////////////////////
// iallocator
template <class Model>
struct iallocator : any::interface<iallocator, Model, any::extends<any::icopyable>>
{
  using iallocator::interface::interface;

  constexpr virtual void *allocate(size_t bytes)
  {
    return any::value(*this).allocate(bytes);
  }

  constexpr virtual void deallocate(void *ptr, size_t bytes) noexcept
  {
    any::value(*this).deallocate(static_cast<std::byte *>(ptr), bytes);
  }
};

using any_allocator = any::any<iallocator>;

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

struct _register_callback_fn;

//////////////////////////////////////////////////////////////////////////////////////////
// istop_token
template <class Model>
struct istop_token
  : any::interface<istop_token, Model, any::extends<any::icopyable, any::iequality_comparable>>
{
  using istop_token::interface::interface;

  constexpr virtual bool stop_requested() const noexcept
  {
    return any::value(*this).stop_requested();
  }
  constexpr virtual bool stop_possible() const noexcept
  {
    return any::value(*this).stop_possible();
  }

private:
  friend struct _register_callback_fn;
  template <class>
  friend struct istop_token;

  constexpr virtual any::any<istop_callback> _register_callback(any::any<icallback> callback)
  {
    return any::value(*this)._register_callback(std::move(callback));
  }
};

//////////////////////////////////////////////////////////////////////////////////////////
// function object for registering a stop callback granted friendship to istop_token
struct _register_callback_fn
{
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
    : token_(token_wrapper(std::move(token)))
  {
  }

  constexpr bool stop_requested() const noexcept
  {
    return token_.stop_requested();
  }

  constexpr bool stop_possible() const noexcept
  {
    return token_.stop_possible();
  }

  template <int = 0, class Token>
  constexpr auto emplace(Token token) -> Token &
  {
    return token_.emplace(token_wrapper(std::move(token)));
  }

  template <class Token, class... Args>
  constexpr auto emplace(Args &&...args) -> Token &
  {
    return token_.template emplace<token_wrapper<Token>>(std::forward<Args>(args)...);
  }

private:
  ////////////////////////////////////////////////////////////////////////////////////////
  // adds to Token a factory function for registering a stop callback
  template <class Token>
  struct token_wrapper : Token
  {
    constexpr token_wrapper(Token token)
      : Token(std::move(token))
    {
    }

    template <class... Args>
    constexpr explicit token_wrapper(Args &&...args)
      : Token(std::forward<Args>(args)...)
    {
    }

    constexpr any::any<istop_callback> _register_callback(any::any<icallback> callback)
    {
      using callback_t = any::_mcall<_stop_callback_for<Token>, any::any<icallback>>;
      Token &token     = *this;
      return any::any<istop_callback>(std::in_place_type<callback_t>, token, std::move(callback));
    }
  };

  struct _callback
  {
    explicit _callback(any_stop_token &token, any::any<icallback> callback)
    {
      callback_.emplace(any::_emplace_from{_register_callback_fn{token.token_, callback}});
    }

  private:
    any::any<istop_callback> callback_{};
  };

  any::any<istop_token> token_;
};

//////////////////////////////////////////////////////////////////////////////////////////
// get_queries
struct get_queries_t
{
  template <detail::_queryable_with<get_queries_t> Queryable>
  static constexpr auto operator()(Queryable const &q)
      -> detail::_query_result_t<Queryable, get_queries_t>
  {
    return q.query(get_queries_t{});
  }
};

inline constexpr auto get_queries = get_queries_t{};

//////////////////////////////////////////////////////////////////////////////////////////
// get_allocator
struct get_allocator_t
{
  using value_type = any_allocator;

  template <detail::_queryable_with<get_allocator_t> Queryable>
  static constexpr auto operator()(Queryable const &q)
      -> detail::_query_result_t<Queryable, get_allocator_t>
  {
    return q.query(get_allocator_t{});
  }
};

inline constexpr get_allocator_t get_allocator{};

//////////////////////////////////////////////////////////////////////////////////////////
// get_scheduler
struct get_scheduler_t
{
  template <detail::_queryable_with<get_scheduler_t> Queryable>
  static constexpr auto operator()(Queryable const &q)
      -> detail::_query_result_t<Queryable, get_scheduler_t>
  {
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
  static constexpr auto operator()(Queryable const &q)
      -> detail::_query_result_t<Queryable, get_stop_token_t>
  {
    return q.query(get_stop_token_t{});
  }
};

inline constexpr auto get_stop_token = get_stop_token_t{};

ANY_DIAG_POP
