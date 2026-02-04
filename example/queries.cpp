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

#include "queries.hpp"

#include <cassert>
#include <iostream>
#include <stop_token>

ANY_DIAG_PUSH
ANY_DIAG_SUPPRESS_CLANG("-Wmissing-braces")

namespace detail
{
template <class Query>
struct _value_type_of
{
  using type = any::any<any::icopyable>;
};

template <class Query>
  requires requires { typename Query::value_type; }
struct _value_type_of<Query>
{
  using type = Query::value_type;
};
} // namespace detail

template <class Query>
using value_type_of_t = detail::_value_type_of<Query>::type;

template <class Query>
inline constexpr bool is_nothrow_query = false;

template <class Query>
  requires requires { Query::is_nothrow_query ? true : false; }
inline constexpr bool is_nothrow_query<Query> = Query::is_nothrow_query;

template <class Model>
struct iqueryable : any::interface<iqueryable, Model>
{
  using iqueryable::interface::interface;

  constexpr virtual bool _try_query(any::type_index type, void *out) const
  {
    return any::value(*this)._try_query(type, out);
  }
};

struct any_queryable
{
public:
  template <class Queryable, class... Queries>
  constexpr explicit any_queryable(Queryable queryable, Queries... queries)
    : queryable_(_dynamically_queryable(std::move(queryable), std::move(queries)...))
  {
  }

  template <class Query>
  constexpr auto query(Query) const noexcept(is_nothrow_query<Query>) -> value_type_of_t<Query>
  {
    value_type_of_t<Query> value;
    queryable_._try_query(any::type_index_of<Query>, std::addressof(value));
    return value;
  }

private:
  friend struct any::access;

  bool _empty_() const noexcept
  {
    return any::empty(queryable_);
  }

  template <class Queryable>
  struct _dynamically_queryable_base
  {
    constexpr auto _mk_try_query(any::type_index type, void *ptr) const
    {
      return [=, this]<class Query>(Query) noexcept(is_nothrow_query<Query>)
      {
        if constexpr (any::_callable_with<Query, Queryable const &>)
        {
          // std::cout << "Type is queryable.\n";
          static_assert(!is_nothrow_query<Query> || noexcept(Query{}(queryable_)),
                        "Queryable::query must be noexcept if Query::is_nothrow_query is true");
          if (type == any::type_index_of<Query>)
          {
            auto &out = *static_cast<value_type_of_t<Query> *>(ptr);
            out.emplace(Query{}(queryable_));
            return true;
          }
        }
        return false;
      };
    }

    Queryable queryable_;
  };

  template <class Queryable, class... Queries>
  struct _dynamically_queryable : _dynamically_queryable_base<Queryable>
  {
    constexpr explicit _dynamically_queryable(Queryable queryable, Queries...)
      : _dynamically_queryable_base<Queryable>{std::move(queryable)}
    {
    }

    constexpr bool _try_query(any::type_index type, void *out) const noexcept
    {
      [[maybe_unused]]
      auto qs         = detail::_with_default{get_queries, any::_mmake_set<>{}}(this->queryable_);
      using queries_t = any::_mapply<any::_mquote<any::_mlist>, decltype(qs), Queries...>;
      return []<class... Qs>(auto fn, any::_mlist<Qs...> *)
      { return (fn(Qs{}) || ...); }(this->_mk_try_query(type, out), (queries_t *)nullptr);
    }
  };

  any::any<iqueryable> queryable_;
};

namespace my
{
struct scheduler
{
};

template <class Query, class Value>
struct prop
{
  constexpr auto query(Query) const -> Value const &
  {
    return value_;
  }

  static constexpr auto query(get_queries_t)
  {
    return any::_mmake_set<Query>();
  }

  [[no_unique_address]] Query query_;
  [[no_unique_address]] Value value_;
};

template <class Query, class Value>
prop(Query, Value) -> prop<Query, Value>;

//////////////////////////////////////////////////////////////////////
// env
template <class... Envs>
struct env;

template <>
struct env<>
{
  constexpr auto query(get_queries_t) const noexcept
  {
    return any::_mmake_set<>();
  }
};

template <class Env>
struct env<Env> : Env
{
};

template <class Env>
struct env<Env &>
{
  template <class Query>
    requires detail::_queryable_with<Env, Query>
  [[nodiscard]]
  constexpr auto query(Query) const noexcept(detail::_nothrow_queryable_with<Env, Query>)
      -> detail::_query_result_t<Env, Query>
  {
    return env_.query(Query{});
  }

  Env &env_;
};

template <class Env1, class Env2>
struct env<Env1, Env2>
{
  template <class Query>
    requires detail::_queryable_with<Env1, Query>
  [[nodiscard]]
  constexpr auto query(Query) const noexcept(detail::_nothrow_queryable_with<Env1, Query>)
      -> detail::_query_result_t<Env1, Query>
  {
    return env1_.query(Query{});
  }

  template <class Query>
    requires detail::_queryable_with<Env1, Query> || detail::_queryable_with<Env2, Query>
  [[nodiscard]]
  constexpr auto query(Query) const noexcept(detail::_nothrow_queryable_with<Env2, Query>)
      -> detail::_query_result_t<Env2, Query>
  {
    return env2_.query(Query{});
  }

  constexpr auto query(get_queries_t) const noexcept = delete;
  constexpr auto query(get_queries_t) const noexcept
    requires detail::_queryable_with<Env1, get_queries_t>
          && detail::_queryable_with<Env2, get_queries_t>
  {
    return env1_.query(get_queries_t{}) + env2_.query(get_queries_t{});
  }

  [[no_unique_address]] Env1 env1_;
  [[no_unique_address]] Env2 env2_;
};

template <class Env1, class Env2, class... Envs>
struct env<Env1, Env2, Envs...> : env<env<Env1, Env2>, Envs...>
{
};

template <class... _Envs>
env(_Envs...) -> env<std::unwrap_reference_t<_Envs>...>;
} // namespace my

struct callback
{
  void operator()() noexcept
  {
  }
};

int main()
{
  auto env = my::env{my::prop{get_allocator, std::allocator<std::byte>()},
                     my::prop{get_scheduler, my::scheduler{}},
                     my::prop{get_stop_token, std::stop_token{}}};

  any_queryable a{env};
  assert(!any::empty(a));

  auto alloc = get_allocator(a);
  static_assert(std::same_as<decltype(alloc), any_allocator>);
  void *ptr = alloc.allocate(128);
  std::cout << "Allocated 128 bytes at " << ptr << "\n";
  alloc.deallocate(ptr, 128);

  auto token = get_stop_token(a);
  static_assert(std::same_as<decltype(token), any_stop_token>);
  any_stop_token::callback_type<callback> cb{token, callback{}};
}

ANY_DIAG_POP
