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

#include "env.hpp"
#include "queries.hpp"

#include <algorithm>
#include <iostream>
#include <stop_token>

ANY_DIAG_PUSH
ANY_DIAG_SUPPRESS_CLANG("-Wmissing-braces")

struct any_queryable
{
public:
  template <class Queryable, class... Queries>
  constexpr explicit any_queryable(Queryable queryable, Queries...)
    : queryable_(_try_queryable<Queryable, Queries...>{{std::move(queryable)}})
  {
  }

  template <class Query>
  [[nodiscard]]
  constexpr auto query(Query) const noexcept(is_always_nothrow_query<Query>) -> query_value_t<Query>
  {
    query_value_t<Query> value;
    queryable_._try_query(any::type_index_of<Query>, std::addressof(value));
    return value;
  }

  template <class Query>
  [[nodiscard]]
  constexpr bool has_query(Query) const noexcept
  {
    return std::ranges::contains(query(get_queries), any::type_index_of<Query>);
  }

  [[nodiscard]]
  static constexpr bool has_query(get_queries_t) noexcept
  {
    return true;
  }

private:
  friend struct any::access; // so that any::empty can call _empty_()

  template <class Model>
  struct iqueryable : any::interface<iqueryable, Model>
  {
    using iqueryable::interface::interface;

    constexpr virtual bool _try_query(any::type_index type, void *out) const
    {
      return any::value(*this)._try_query(type, out);
    }
  };

  template <class Queryable>
  struct _try_queryable_base
  {
    [[nodiscard]]
    constexpr auto _mk_try_query(any::type_index type, void *ptr) const
    {
      // BUGBUG this can only be nothrow if the emplace/assign is nothrow:
      return [=, this]<class Query>(Query) noexcept(is_always_nothrow_query<Query>)
      {
        if constexpr (any::_callable_with<Query, Queryable const &>)
        {
          static_assert(!is_always_nothrow_query<Query> || noexcept(Query()(queryable_)),
                        "Queryable::query must be noexcept if Query::always_nothrow is true");
          if (type == any::type_index_of<Query>)
          {
            auto &out = *static_cast<query_value_t<Query> *>(ptr);
            // emplace if possible, otherwise assign:
            if constexpr (requires { out.emplace(Query()(queryable_)); })
              out.emplace(Query()(queryable_));
            else
              out = Query()(queryable_);
            return true;
          }
        }
        return false;
      };
    }

    Queryable queryable_;
  };

  template <class Queryable, class... Queries>
  struct _try_queryable : _try_queryable_base<Queryable>
  {
    constexpr bool _try_query(any::type_index type, void *out) const noexcept
    {
      // Handle get_queries specially
      if (type == any::type_index_of<get_queries_t>)
      {
        auto &queries = *static_cast<query_value_t<get_queries_t> *>(out);
        queries       = _all_queries_t::value;
        return true;
      }
      else
      {
        // Turn the query set into a type list containing all the supported queries
        using all_queries_t = any::_mapply<any::_mquote<any::_mlist>, _all_queries_t> *;
        return _try_queries(this->_mk_try_query(type, out), all_queries_t());
      }
    }

  private:
    // The set of supported queries is the union of Queries... and those supported by
    // Queryable (if known)
    using _get_queries_fn_t   = detail::_with_default<get_queries_t, query_set<>>;
    using _implicit_queries_t = any::_call_result_t<_get_queries_fn_t, Queryable const &>;
    // Unpack all the queries into a query_set to make them unique:
    using _all_queries_t =
        any::_mapply<any::_mquote<query_set>, _implicit_queries_t, Queries..., get_queries_t>;
  };

  [[nodiscard]]
  constexpr bool _empty_() const noexcept
  {
    return any::empty(queryable_);
  }

  template <class Fn, class... Qs>
  static constexpr bool _try_queries(Fn fn, any::_mlist<Qs...> *) noexcept
  {
    return (fn(Qs()) || ...);
  }

  any::any<iqueryable> queryable_;
};

namespace my
{
struct scheduler
{
};

struct callback
{
  void operator()() noexcept
  {
  }
};
} // namespace my

int main()
{
  auto env = ::env{prop{get_allocator, std::allocator<std::byte>()},
                   prop{get_scheduler, my::scheduler{}},
                   prop{get_stop_token, std::stop_token{}}};

  any_queryable a{env};
  assert(!any::empty(a));
  assert(a.has_query(get_scheduler));
  assert(!a.has_query('?'));
  for (auto type : get_queries(a))
  {
    std::cout << "Supports query: " << type.name() << "\n";
  }

  auto alloc = get_allocator(a);
  static_assert(std::same_as<decltype(alloc), any_allocator<std::byte>>);
  std::byte *ptr = alloc.allocate(128);
  std::cout << "Allocated 128 bytes at " << ptr << "\n";
  alloc.deallocate(ptr, 128);

  auto token = get_stop_token(a);
  static_assert(std::same_as<decltype(token), any_stop_token>);
  any_stop_token::callback_type<my::callback> cb{token, my::callback{}};
}

ANY_DIAG_POP
