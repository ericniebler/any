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

#include "queries.hpp"

#include <type_traits> // For std::unwrap_reference_t

template <class Query, class Value>
struct prop
{
  constexpr auto query(Query) const -> Value const &
  {
    return value_;
  }

  static constexpr auto query(get_queries_t)
  {
    return query_set<Query>();
  }

  [[no_unique_address]] Query query_;
  [[no_unique_address]] Value value_;
};

//////////////////////////////////////////////////////////////////////
// env
template <class... Envs>
struct env;

template <>
struct env<>
{
  constexpr auto query(get_queries_t) const noexcept
  {
    return query_set<>();
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
  // Prefer Env1 when both environments can answer the query:
  template <class Query>
    requires detail::_queryable_with<Env1, Query>
  [[nodiscard]]
  constexpr auto query(Query) const noexcept(detail::_nothrow_queryable_with<Env1, Query>)
      -> detail::_query_result_t<Env1, Query>
  {
    return env1_.query(Query{});
  }

  // Use Env2 when Env1 cannot answer the query:
  template <class Query>
    requires detail::_queryable_with<Env1, Query> || detail::_queryable_with<Env2, Query>
  [[nodiscard]]
  constexpr auto query(Query) const noexcept(detail::_nothrow_queryable_with<Env2, Query>)
      -> detail::_query_result_t<Env2, Query>
  {
    return env2_.query(Query{});
  }

  // Provide the combined query set, but only if both environments support get_queries:
  constexpr auto query(get_queries_t) const noexcept = delete;
  constexpr auto query(get_queries_t) const noexcept
    requires detail::_queryable_with<Env1, get_queries_t>
          && detail::_queryable_with<Env2, get_queries_t>
  {
    // Combine the query sets from both environments:
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
