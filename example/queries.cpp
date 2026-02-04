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

#include <cassert>
#include <iostream>
#include <stop_token>

ANY_DIAG_PUSH
ANY_DIAG_SUPPRESS_CLANG("-Wmissing-braces")

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
} // namespace my

struct callback
{
  void operator()() noexcept
  {
  }
};

int main()
{
  auto env = ::env{prop{get_allocator, std::allocator<std::byte>()},
                   prop{get_scheduler, my::scheduler{}},
                   prop{get_stop_token, std::stop_token{}}};

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
