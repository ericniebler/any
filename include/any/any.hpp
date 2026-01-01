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

#include "detail/config.hpp"
#include "detail/typeinfo.hpp"
#include "detail/utility.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <exception>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

ANY_DIAG_PUSH
ANY_DIAG_SUPPRESS_MSVC(4141) // 'inline' used more than once

//////////////////////////////////////////////////////////////////////////////////////////
//! any: a library for ad hoc polymorphism with value semantics
//!
//! @par Terminology:
//!
//! - "root":
//!
//!   A type satisfying the @c root concept that is used as the nucleus of a "model".
//!   There are 5 root types:
//!
//!   - @c _iroot:                 the abstract root
//!   - @c _value_root:            holds a concrete value
//!   - @c _reference_root:        holds a concrete reference
//!   - @c _value_proxy_root:      holds a type-erased value model
//!   - @c _reference_proxy_root:  holds a type-erased reference model
//!
//!   Aside from @c _iroot, all root types inherit from @c iabstract<Interface>, where
//!   @c Interface is the interface that the root type implements.
//!
//!   The @c root concept is defined as:
//!
//!   @code
//!   template <class Root>
//!   concept root = requires (Root& root)
//!   {
//!     any::value(root);
//!     { any::reset(root); } -> std::same_as<void>;
//!     { any::type(root) } -> std::same_as<const type_info &>;
//!     { any::data(root) } -> std::same_as<void *>;
//!     { any::empty(root) } -> std::same_as<bool>;
//!   };
//!   @endcode
//!
//! - "model":
//!
//!   A polymorphic wrapper around a root that is constructed by recursively applying a
//!   given interface and its base interfaces to the root type. For example, given an
//!   interface @c Derived that extends @c Base, the value proxy model is a type derived
//!   from @c Derived<Base<_value_proxy_root<Derived>>>. Model types implement their given
//!   interfaces in terms of the root type. There are 5 model types:
//!
//!   - @c iabstract:              akin to an abstract base class for the
//!                                interface
//!   - @c _value_model:           implements the interface for a concrete value
//!   - @c _reference_model:       implements the interface for a concrete
//!                                reference
//!   - @c _value_proxy_model:     implements the interface over a type-erased
//!                                value model
//!   - @c _reference_proxy_model: implements the interface over a type-erased
//!                                reference model
//!
//! - "proxy":
//!
//!   A level of indirection that stores either a type-erased model in a small buffer or a
//!   pointer to an object stored elsewhere. The @c _value_proxy_root and @c
//!   _reference_proxy_root types model the @c root concept and contain an array of bytes
//!   in which they stores either a polymorphic model in-situ or a (tagged) pointer to a
//!   heap-allocated model. The @c _value_proxy_model and @c _reference_proxy_model types
//!   implement the given interface in terms of the root type.
//!
//! @par Notes:
//!
//! - @c Interface<Base> inherits directly from @c any::interface<Interface,Base>, which
//!   inherits directly from @c Base.
//!
//! - Given an interface template @c Derived that extends @c Base, the type
//!   @c iabstract<Derived> is derived from @c iabstract<Base>.
//!
//! - In the case of multiple interface extension, the inheritance is forced to be linear.
//!   As a result, for an interface @c C that extends @c A and @c B (in that order),
//!   @c iabstract<C> will have a linear inheritance hierarchy; it will be an alias for
//!   @c C<B<A<_iroot>>>. The result is that @c iabstract<C> inherits from @c iabstract<A>
//!   but not from @c iabstract<B>.
//!
//! - The "`_proxy_root`" types both implement an @c emplace function that accepts a
//!   concrete value or reference, wraps it in the appropriate "`_model`" type, and stores
//!   it either in-situ or on the heap depending on its size and whether it is nothrow
//!   moveable.
//!
//! - The @c _root types (excluding @c _iroot) all inherit from @c iabstract<Interface>.
//!   The @c _model types implement the interface in terms of the root type.
//!
//! - @c any<Derived> inherits from @c _value_proxy_model<Derived>, which in turn inherits
//!   from @c Derived<Base<_value_proxy_root<Derived>>>, which in turn inherits from
//!   @c Derived<Base<_iroot>> (aka @c iabstract<Derived> ).
//!
//! - @c any_ptr<Derived> is implemented in terms of a mutable private
//!   @c _reference_proxy_model<Derived> data member, which in turn inherits from
//!   @c Derived<Base<_reference_proxy_root<Derived>>>.
//!
//! - For every @c any<Interface> instantiation, there are 5 instantiations of
//!   @c Interface:
//!
//!   1. @c Interface<...Bases...<_iroot>>>...>
//!   2. @c Interface<...Bases...<_value_root<Value,Interface>>...>
//!   3. @c Interface<...Bases...<_reference_root<Value,Interface>>...>
//!   4. @c Interface<...Bases...<_value_proxy_root<Interface>>...>
//!   5. @c Interface<...Bases...<_reference_proxy_root<Interface>>...>

namespace any
{
constexpr size_t default_buffer_size = 3 * sizeof(void *);
constexpr char const *_pure_virt_msg = "internal error: pure virtual %s() called\n";

//////////////////////////////////////////////////////////////////////////////////////////
// forward declarations

// any types
template <template <class> class Interface>
struct any;

template <template <class> class Interface>
struct _any_ptr_base;

template <template <class> class Interface>
struct any_ptr;

template <template <class> class Interface>
struct any_const_ptr;

template <template <class> class... BaseInterfaces>
struct extends;

// semiregular interfaces
template <class Base>
struct imovable;

template <class Base>
struct icopyable;

template <class Base>
struct iequality_comparable;

template <class Base>
struct isemiregular;

struct _iroot;

template <template <class> class Interface>
using _bases_of = Interface<_iroot>::_bases_type;

template <template <class> class Interface,
          class Base,
          class BaseInterfaces = extends<>,
          size_t BufferSize    = default_buffer_size>
struct interface;

//////////////////////////////////////////////////////////////////////////////////////////
// interface_cast
template <template <class> class Interface, class Base>
[[ANY_ALWAYS_INLINE, nodiscard]]
inline constexpr Interface<Base> &interface_cast(Interface<Base> &iface) noexcept
{
  return iface;
}

template <template <class> class Interface, class Base>
[[ANY_ALWAYS_INLINE, nodiscard]]
inline constexpr Interface<Base> const &interface_cast(Interface<Base> const &iface) noexcept
{
  return iface;
}

//////////////////////////////////////////////////////////////////////////////////////////
// accessors
struct _access
{
  struct _value_t
  {
    template <class T>
    [[ANY_ALWAYS_INLINE, nodiscard]]
    inline constexpr auto &operator()(T &&t) const noexcept
    {
      return std::forward<T>(t)._value_();
    }
  };

  struct _empty_t
  {
    template <class T>
    [[ANY_ALWAYS_INLINE, nodiscard]]
    inline constexpr bool operator()(T const &t) const noexcept
    {
      return t._empty_();
    }
  };

  struct _reset_t
  {
    template <class T>
    [[ANY_ALWAYS_INLINE]]
    inline constexpr void operator()(T &t) const noexcept
    {
      t._reset_();
    }
  };

  struct _type_t
  {
    template <class T>
    [[ANY_ALWAYS_INLINE, nodiscard]]
    inline constexpr type_info const &operator()(T const &t) const noexcept
    {
      return t._type_();
    }
  };

  struct _data_t
  {
    template <class T>
    [[ANY_ALWAYS_INLINE, nodiscard]]
    inline constexpr auto *operator()(T &t) const noexcept
    {
      return t._data_();
    }
  };

  struct caddressof_t
  {
    template <template <class> class Interface, class Base>
    [[nodiscard]]
    constexpr auto operator()(Interface<Base> const &iface) const noexcept
    {
      return any_const_ptr<Interface>(std::addressof(iface));
    }
  };

  struct addressof_t : caddressof_t
  {
    using caddressof_t::operator();

    template <template <class> class Interface, class Base>
    [[nodiscard]]
    constexpr auto operator()(Interface<Base> &iface) const noexcept
    {
      return any_ptr<Interface>(std::addressof(iface));
    }
  };
};

[[maybe_unused]] inline constexpr auto value      = _access::_value_t{};
[[maybe_unused]] inline constexpr auto empty      = _access::_empty_t{};
[[maybe_unused]] inline constexpr auto reset      = _access::_reset_t{};
[[maybe_unused]] inline constexpr auto type       = _access::_type_t{};
[[maybe_unused]] inline constexpr auto data       = _access::_data_t{};
[[maybe_unused]] inline constexpr auto addressof  = _access::addressof_t{};
[[maybe_unused]] inline constexpr auto caddressof = _access::caddressof_t{};

// value_of_t
template <class T>
using value_of_t = std::decay_t<decltype(value(std::declval<T &>()))>;

//////////////////////////////////////////////////////////////////////////////////////////
// extension_of
template <class Interface, template <class> class BaseInterface>
concept extension_of =
    requires(Interface const &iface) { ::any::interface_cast<BaseInterface>(iface); };

//////////////////////////////////////////////////////////////////////////////////////////
// _is_small: Model is Interface<T> for some concrete T
template <class Model>
[[nodiscard]]
constexpr bool _is_small(size_t buffer_size) noexcept
{
  constexpr bool nothrow_movable =
      !extension_of<Model, imovable> || std::is_nothrow_move_constructible_v<Model>;
  return sizeof(Model) <= buffer_size && nothrow_movable;
}

//////////////////////////////////////////////////////////////////////////////////////////
// _tagged_ptr
struct _tagged_ptr
{
  [[ANY_ALWAYS_INLINE]]
  /*implicit*/ constexpr inline _tagged_ptr() noexcept
    : data_(std::uintptr_t(1))
  {
  }

  [[ANY_ALWAYS_INLINE]]
  /*implicit*/ inline _tagged_ptr(void *ptr) noexcept
    : data_(reinterpret_cast<std::uintptr_t>(ptr) | std::uintptr_t(1))
  {
  }

  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline void *_get() const noexcept
  {
    ANY_ASSERT(!_is_vptr());
    return reinterpret_cast<void *>(data_ & ~std::uintptr_t(1));
  }

  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline constexpr bool _is_vptr() const noexcept
  {
    return (data_ & 1) == 0;
  }

  [[nodiscard]]
  constexpr bool operator==(std::nullptr_t) const noexcept
  {
    return data_ == 1;
  }

  std::uintptr_t data_;
};

// template <class Root>
// concept root = requires (Root& root)
// {
//   root._value_();
//   root._reset_();
//   { root._type_() } -> std::same_as<const type_info &>;
//   { root._data_() } -> std::same_as<void *>;
//   { root._empty_() } -> std::same_as<bool>;
// };

//! @c iabstract must be an alias in order for @c iabstract<Derived> to be
//! derived from
//! @c iabstract<Base>. @c iabstract<Derived> is an alias for @c
//! Derived<Base<_iroot>>.
template <template <class> class Interface, class BaseInterfaces = _bases_of<Interface>>
using iabstract = Interface<_mcall<BaseInterfaces, _iroot>>;

enum class _box_kind
{
  _abstract,
  _object,
  _proxy
};

enum class _root_kind
{
  _value,
  _reference
};

//////////////////////////////////////////////////////////////////////////////////////////
// _iroot
struct _iroot
{
  static constexpr ::any::_box_kind _box_kind = ::any::_box_kind::_abstract;
  static constexpr size_t _buffer_size        = sizeof(_tagged_ptr); // minimum size
  using _bases_type                           = extends<>;

  // needed by MSVC for EBO to work for some reason:
  constexpr virtual ~_iroot() = default;

private:
  template <template <class> class, class, class, size_t>
  friend struct interface;
  friend struct _access;

  template <class Self>
  constexpr Self &&_value_(this Self &&) noexcept
  {
    return ::any::_die<Self &&>(_pure_virt_msg, "value");
  }

  [[nodiscard]]
  constexpr virtual bool _empty_() const noexcept
  {
    return ::any::_die<bool>(_pure_virt_msg, "empty");
  }

  constexpr virtual void _reset_() noexcept
  {
    ::any::_die(_pure_virt_msg, "reset");
  }

  [[nodiscard]]
  constexpr virtual type_info const &_type_() const noexcept
  {
    return ::any::_die<type_info const &>(_pure_virt_msg, "type");
  }

  [[nodiscard]]
  constexpr virtual void *_data_() const noexcept
  {
    return ::any::_die<void *>(_pure_virt_msg, "data");
  }

  // public:
  void _slice_to_() noexcept            = delete;
  void _indirect_bind_() const noexcept = delete;
};

//////////////////////////////////////////////////////////////////////////////////////////
// _box
template <template <class> class Interface, class Value>
struct _box : iabstract<Interface>
{
  constexpr explicit _box(Value &&value) noexcept
    : value_(std::move(value))
  {
  }

  template <class Self>
  [[nodiscard]]
  constexpr auto &&_value_(this Self &&self) noexcept
  {
    return static_cast<Self &&>(self).value_;
  }

private:
  Value value_;
};

// // A specialization of _box to take advantage of EBO (empty base optimization):
template <template <class> class Interface, class Value>
  requires std::is_empty_v<Value> && (!std::is_final_v<Value>)
struct [[ANY_EMPTY_BASES]] _box<Interface, Value>
  : iabstract<Interface>
  , private Value
{
  constexpr explicit _box(Value &&value) noexcept
    : Value(std::move(value))
  {
  }

  template <class Self>
  [[nodiscard]]
  constexpr auto &&_value_(this Self &&self) noexcept
  {
    return std::forward<_copy_cvref_t<Self, Value>>(self);
  }
};

template <class Interface, _box_kind BoxKind>
concept _has_box_kind = std::remove_reference_t<Interface>::_box_kind == BoxKind;

// Without the check against _has_box_kind, this concept would always be
// satisfied when building an object model or a proxy model because of the
// abstract implementation of Interface in the iabstract layer.
//
// any<Derived>
//   : _value_proxy_model<Derived, V>
//       : Derived<Base<_value_proxy_root<Derived, V>>>    // box_kind == proxy
//         ^^^^^^^        : Derived<Base<_iroot>>          // box_kind == abstract
//                          ^^^^^^^

template <class Derived, template <class> class Interface>
concept _already_implements = requires(Derived const &iface) {
  { ::any::interface_cast<Interface>(iface) } -> _has_box_kind<Derived::_box_kind>;
};

// If we are slicing into a buffer that is smaller than our own, then slicing
// may throw.
template <class Interface, class Base, size_t BufferSize>
concept _nothrow_slice = (Base::_box_kind != _box_kind::_abstract)
                      && (Base::_root_kind == _root_kind::_value)
                      && (Interface::_buffer_size >= BufferSize);

//////////////////////////////////////////////////////////////////////////////////////////
// extends
template <>
struct extends<>
{
  template <class Base>
  using call = Base;
};

template <template <class> class BaseInterface, template <class> class... BaseInterfaces>
struct extends<BaseInterface, BaseInterfaces...>
{
  template <class Base, class BasesOfBase = _mcall<_bases_of<BaseInterface>, Base>>
  using call = _mcall<
      extends<BaseInterfaces...>,
      // If Base already implements BaseInterface, do not re-apply it.
      _if_t<_already_implements<Base, BaseInterface>, BasesOfBase, BaseInterface<BasesOfBase>>>;
};

//////////////////////////////////////////////////////////////////////////////////////////
// _emplace_into
template <class Model, class... Args>
constexpr Model &_emplace_into([[maybe_unused]] _iroot *&pointer,
                               [[maybe_unused]] std::span<std::byte> buffer,
                               Args &&...args)
{
  static_assert(_decayed<Model>);
  if consteval
  {
    pointer = ::new Model(std::forward<Args>(args)...);
    return *static_cast<Model *>(pointer);
  }
  else
  {
    if (::any::_is_small<Model>(buffer.size()))
    {
      return *std::construct_at(reinterpret_cast<Model *>(buffer.data()),
                                std::forward<Args>(args)...);
    }
    else
    {
      auto *const model = ::new Model(std::forward<Args>(args)...);
      *::any::start_lifetime_as<_tagged_ptr>(buffer.data()) = _tagged_ptr(model);
      return *model;
    }
  }
}

template <int = 0, class CvRefValue, class Value = std::decay_t<CvRefValue>>
[[ANY_ALWAYS_INLINE]]
inline constexpr Value &
_emplace_into(_iroot *&pointer, std::span<std::byte> buffer, CvRefValue &&value)
{
  return ::any::_emplace_into<Value>(pointer, buffer, std::forward<CvRefValue>(value));
}

// reference
template <template <class> class Interface, class CvValue, class Extension = iabstract<Interface>>
struct _reference_root;

template <template <class> class Interface, class CvValue, class Extension = iabstract<Interface>>
struct _reference_model
  : Interface<_mcall<_bases_of<Interface>, _reference_root<Interface, CvValue, Extension>>>
{
  using _base_t =
      Interface<_mcall<_bases_of<Interface>, _reference_root<Interface, CvValue, Extension>>>;
  using _base_t::_base_t;
};

// reference proxy
template <template <class> class Interface>
struct _reference_proxy_root;

template <template <class> class Interface>
struct _reference // _reference_proxy_model
  : Interface<_mcall<_bases_of<Interface>, _reference_proxy_root<Interface>>>
{
};

template <template <class> class Interface>
using _reference_proxy_model = _reference<Interface>;

// value
template <template <class> class Interface, class Value>
struct _value_root;

template <template <class> class Interface, class Value>
struct _value_model final : Interface<_mcall<_bases_of<Interface>, _value_root<Interface, Value>>>
{
  using _base_t = Interface<_mcall<_bases_of<Interface>, _value_root<Interface, Value>>>;
  using _base_t::_base_t;

  // This is a virtual override if Interface extends imovable
  //! @pre ::any::_is_small<_value_model>(buffer.size())
  constexpr void _move_to(_iroot *&pointer, std::span<std::byte> buffer) noexcept
  {
    static_assert(extension_of<iabstract<Interface>, imovable>);
    ANY_ASSERT(::any::_is_small<_value_model>(buffer.size()));
    ::any::_emplace_into(pointer, buffer, std::move(*this));
    reset(*this);
  }

  // This is a virtual override if Interface extends icopyable
  constexpr void _copy_to(_iroot *&pointer, std::span<std::byte> buffer) const
  {
    static_assert(extension_of<iabstract<Interface>, icopyable>);
    ANY_ASSERT(!empty(*this));
    ::any::_emplace_into(pointer, buffer, *this);
  }
};

// value proxy
template <template <class> class Interface>
struct _value_proxy_root;

template <template <class> class Interface>
struct _value // _value_proxy_model
  : Interface<_mcall<_bases_of<Interface>, _value_proxy_root<Interface>>>
{
};

template <template <class> class Interface>
using _value_proxy_model = _value<Interface>;

//////////////////////////////////////////////////////////////////////////////////////////
//! interface
template <template <class> class Interface, class Base, class BaseInterfaces, size_t BufferSize>
struct interface : Base
{
  using _bases_type     = BaseInterfaces;
  using _interface_type = iabstract<Interface, BaseInterfaces>;
  using Base::_indirect_bind_;
  using Base::_slice_to_;
  using Base::Base;

  static constexpr size_t _buffer_size =
      BufferSize > Base::_buffer_size ? BufferSize : Base::_buffer_size;

  static constexpr bool _nothrow_slice = ::any::_nothrow_slice<_interface_type, Base, _buffer_size>;

  //! @pre !empty(*this)
  constexpr virtual void _slice_to_(_value_proxy_root<Interface> &) noexcept(_nothrow_slice)
  {
    ::any::_die(_pure_virt_msg, "slice_to");
  }

  //! @pre !empty(*this)
  constexpr virtual void _indirect_bind_(_reference_proxy_root<Interface> &) noexcept
  {
    ::any::_die(_pure_virt_msg, "_indirect_bind_");
  }

  //! @pre !empty(*this)
  constexpr virtual void _indirect_bind_(_reference_proxy_root<Interface> &) const noexcept
  {
    ::any::_die(_pure_virt_msg, "_indirect_bind_");
  }
};

template <template <class> class Interface, class Base, class BaseInterfaces, size_t BufferSize>
  requires(Base::_box_kind == _box_kind::_proxy)
struct interface<Interface, Base, BaseInterfaces, BufferSize> : Base
{
  using _bases_type     = BaseInterfaces;
  using _interface_type = iabstract<Interface, BaseInterfaces>;
  using Base::_indirect_bind_;
  using Base::_slice_to_;
  using Base::Base;

  static constexpr size_t _buffer_size =
      BufferSize > Base::_buffer_size ? BufferSize : Base::_buffer_size;

  static constexpr bool _nothrow_slice = ::any::_nothrow_slice<_interface_type, Base, _buffer_size>;

  //! @pre !empty(*this)
  template <class...> // template so it is not considered an override
  constexpr void _slice_to_(_value_proxy_root<Interface> &out) noexcept(_nothrow_slice)
  {
    ANY_ASSERT(!empty(*this));
    value(*this)._slice_to_(out);
    reset(*this);
  }

  //! @pre !empty(*this)
  template <class Self>
  constexpr void _indirect_bind_(this Self &self, _reference_proxy_root<Interface> &out) noexcept
  {
    ANY_ASSERT(!empty(self));
    value(self)._indirect_bind_(out);
  }
};

template <template <class> class Interface, class Base, class BaseInterfaces, size_t BufferSize>
  requires(Base::_box_kind == _box_kind::_object)
struct interface<Interface, Base, BaseInterfaces, BufferSize> : Base
{
  using _bases_type     = BaseInterfaces;
  using _interface_type = iabstract<Interface, BaseInterfaces>;
  using Base::_indirect_bind_;
  using Base::_slice_to_;
  using Base::Base;

  static constexpr size_t _buffer_size =
      BufferSize > Base::_buffer_size ? BufferSize : Base::_buffer_size;

  static constexpr bool _nothrow_slice = ::any::_nothrow_slice<_interface_type, Base, _buffer_size>;

  //! @pre !empty(*this)
  constexpr void
  _slice_to_(_value_proxy_root<Interface> &out) noexcept(_nothrow_slice) final override
  {
    ANY_ASSERT(!empty(*this));
    // Move from type-erased values, but not from type-erased references
    constexpr bool is_value = (Base::_root_kind == _root_kind::_value);
    out.emplace(auto(::any::_move_if<is_value>(value(*this)))); // potentially throwing
  }

  //! @pre !empty(*this)
  constexpr void _indirect_bind_(_reference_proxy_root<Interface> &out) noexcept final override
  {
    ANY_ASSERT(!empty(*this));
    out._object_bind_(*this);
  }

  //! @pre !empty(*this)
  constexpr void
  _indirect_bind_(_reference_proxy_root<Interface> &out) const noexcept final override
  {
    ANY_ASSERT(!empty(*this));
    out._object_bind_(*this);
  }
};

//////////////////////////////////////////////////////////////////////////////////////////
// _value_root
template <template <class> class Interface, class Value>
struct _value_root : _box<Interface, Value>
{
  template <class CvSelf>
  using element_type                            = _copy_cvref_t<CvSelf, Value>;
  using interface_type                          = iabstract<Interface>;
  using value_type                              = Value;

  static constexpr ::any::_box_kind _box_kind   = ::any::_box_kind::_object;
  static constexpr ::any::_root_kind _root_kind = ::any::_root_kind::_value;

  constexpr explicit _value_root(Value value) noexcept
    : _box<Interface, Value>(std::move(value))
  {
  }

  [[nodiscard]]
  constexpr bool _empty_() const noexcept final override
  {
    return false;
  }

  constexpr void _reset_() noexcept final override
  {
    // no-op
  }

  [[nodiscard]]
  constexpr type_info const &_type_() const noexcept final override
  {
    return ANY_TYPEID(Value);
  }

  [[nodiscard]]
  constexpr void *_data_() const noexcept final override
  {
    return const_cast<void *>(static_cast<void const *>(std::addressof(value(*this))));
  }
};

//////////////////////////////////////////////////////////////////////////////////////////
// _value_proxy_root
template <template <class> class Interface>
struct _value_proxy_root : iabstract<Interface>
{
  using interface_type                          = iabstract<Interface>;
  static constexpr ::any::_box_kind _box_kind   = ::any::_box_kind::_proxy;
  static constexpr ::any::_root_kind _root_kind = ::any::_root_kind::_value;

  static constexpr bool _movable                = extension_of<iabstract<Interface>, imovable>;
  static constexpr bool _copyable               = extension_of<iabstract<Interface>, icopyable>;

  [[ANY_ALWAYS_INLINE]]
  inline constexpr _value_proxy_root() noexcept
  {
    if consteval
    {
      pointer_ = nullptr;
    }
    else
    {
      *::any::start_lifetime_as<_tagged_ptr>(buffer_) = _tagged_ptr();
    }
  }

  constexpr _value_proxy_root(_value_proxy_root &&other) noexcept
    requires _movable
    : _value_proxy_root()
  {
    swap(other);
  }

  constexpr _value_proxy_root(_value_proxy_root const &other)
    requires _copyable
    : _value_proxy_root()
  {
    if (!empty(other))
      value(other)._copy_to(pointer_, buffer_);
  }

  constexpr ~_value_proxy_root()
  {
    _reset_();
  }

  constexpr _value_proxy_root &operator=(_value_proxy_root &&other) noexcept
    requires _movable
  {
    if (this != &other)
    {
      _reset_();
      swap(other);
    }
    return *this;
  }

  constexpr _value_proxy_root &operator=(_value_proxy_root const &other)
    requires _copyable
  {
    if (this != &other)
      _value_proxy_root(other).swap(*this);
    return *this;
  }

  constexpr void swap(_value_proxy_root &other) noexcept
    requires _movable
  {
    if consteval
    {
      std::swap(pointer_, other.pointer_);
    }
    else
    {
      if (this == &other)
        return;

      auto &this_ptr = *::any::start_lifetime_as<_tagged_ptr>(buffer_);
      auto &that_ptr = *::any::start_lifetime_as<_tagged_ptr>(other.buffer_);

      // This also covers the case where both this_ptr and that_ptr are null.
      if (!this_ptr._is_vptr() && !that_ptr._is_vptr())
        return std::swap(this_ptr, that_ptr);

      if (this_ptr == nullptr)
        return value(other)._move_to(pointer_, buffer_);

      if (that_ptr == nullptr)
        return value(*this)._move_to(other.pointer_, other.buffer_);

      auto temp = std::move(*this);
      value(other)._move_to(pointer_, buffer_);
      value(temp)._move_to(other.pointer_, other.buffer_);
    }
  }

  template <class Value, class... Args>
  constexpr Value &emplace(Args &&...args)
  {
    _reset_();
    return _emplace_<Value>(std::forward<Args>(args)...);
  }

  template <int = 0, class CvRefValue, class Value = std::decay_t<CvRefValue>>
  constexpr Value &emplace(CvRefValue &&value)
  {
    _reset_();
    return _emplace_<Value>(std::forward<CvRefValue>(value));
  }

  [[nodiscard]]
  constexpr bool _in_situ_() const noexcept
  {
    if consteval
    {
      return false;
    }
    else
    {
      return ::any::start_lifetime_as<_tagged_ptr>(buffer_)->_is_vptr();
    }
  }

private:
  template <template <class> class>
  friend struct any;
  friend struct _access;

  template <class Value, class... Args>
  constexpr Value &_emplace_(Args &&...args)
  {
    static_assert(_decayed<Value>, "Value must be an object type.");
    using model_type = _value_model<Interface, Value>;
    auto &model = ::any::_emplace_into<model_type>(pointer_, buffer_, std::forward<Args>(args)...);
    return model._value_();
  }

  template <int = 0, class CvRefValue, class Value = std::decay_t<CvRefValue>>
  constexpr Value &_emplace_(CvRefValue &&value)
  {
    return _emplace_<Value>(std::forward<CvRefValue>(value));
  }

  template <class Self>
  [[nodiscard]]
  constexpr auto &&_value_(this Self &&self) noexcept
  {
    using root_ptr_t      = std::add_pointer_t<_copy_cvref_t<Self, _iroot>>;
    using interface_ref_t = _copy_cvref_t<Self &&, iabstract<Interface>>;
    using interface_ptr_t = std::add_pointer_t<interface_ref_t>;
    if consteval
    {
      return static_cast<interface_ref_t>(
          *::any::_polymorphic_downcast<interface_ptr_t>(self.pointer_));
    }
    else
    {
      auto const ptr = *::any::start_lifetime_as<_tagged_ptr>(self.buffer_);
      ANY_ASSERT(ptr != nullptr);
      auto *root_ptr = static_cast<root_ptr_t>(ptr._is_vptr() ? self.buffer_ : ptr._get());
      return static_cast<interface_ref_t>(*::any::_polymorphic_downcast<interface_ptr_t>(root_ptr));
    }
  }

  [[nodiscard]]
  constexpr bool _empty_() const noexcept final override
  {
    if consteval
    {
      return pointer_ == nullptr;
    }
    else
    {
      return *::any::start_lifetime_as<_tagged_ptr>(buffer_) == nullptr;
    }
  }

  [[ANY_ALWAYS_INLINE]]
  inline constexpr void _reset_() noexcept final override
  {
    if consteval
    {
      delete std::exchange(pointer_, nullptr);
    }
    else
    {
      auto &ptr = *::any::start_lifetime_as<_tagged_ptr>(buffer_);
      if (ptr == nullptr)
        return;
      else if (ptr._is_vptr())
        std::destroy_at(std::addressof(_value_()));
      else
        delete std::addressof(_value_());

      ptr = _tagged_ptr();
    }
  }

  [[nodiscard]]
  constexpr type_info const &_type_() const noexcept final override
  {
    return _empty_() ? ANY_TYPEID(void) : type(_value_());
  }

  [[nodiscard]]
  constexpr void *_data_() const noexcept final override
  {
    return _empty_() ? nullptr : data(_value_());
  }

  union
  {
    _iroot *pointer_ = nullptr;                            //!< Used in consteval context
    std::byte buffer_[iabstract<Interface>::_buffer_size]; //!< Used in runtime context
  };
};

//////////////////////////////////////////////////////////////////////////////////////////
// _reference_root
template <template <class> class Interface, class CvValue>
struct _reference_root<Interface, CvValue> : iabstract<Interface>
{
  static_assert(!extension_of<CvValue, Interface>,
                "Value must be a concrete type, not an Interface type.");

  template <class>
  using element_type                            = CvValue;
  using value_type                              = std::remove_cv_t<CvValue>;
  using interface_type                          = iabstract<Interface>;

  static constexpr ::any::_box_kind _box_kind   = ::any::_box_kind::_object;
  static constexpr ::any::_root_kind _root_kind = ::any::_root_kind::_reference;

  _reference_root()                             = default;

  constexpr explicit _reference_root(CvValue &value) noexcept
    : value_(std::addressof(value))
  {
  }

  constexpr explicit _reference_root(_iroot &root, bool is_reference) noexcept
  {
    _init<Interface>(root, is_reference);
  }

  template <class Self>
  [[nodiscard]]
  constexpr auto &&_value_(this Self &&self) noexcept
  {
    using value_ref_t = _copy_cvref_t<Self &&, std::remove_cv_t<CvValue>>;
    if !consteval
    {
      ANY_ASSERT(
          (std::convertible_to<CvValue &, value_ref_t>)
          && "attempt to get a mutable reference from a const reference, or an rvalue from an "
             "lvalue");
    }
    return static_cast<value_ref_t>(const_cast<value_ref_t &>(*self.value_));
  }

  [[nodiscard]]
  constexpr bool _empty_() const noexcept final override
  {
    return false;
  }

  constexpr void _reset_() noexcept final override
  {
    // no-op
  }

  [[nodiscard]]
  constexpr type_info const &_type_() const noexcept final override
  {
    return ANY_TYPEID(value_type);
  }

  [[nodiscard]]
  constexpr void *_data_() const noexcept final override
  {
    return const_cast<void *>(static_cast<void const *>(value_));
  }

private:
  template <template <class> class, class, class>
  friend struct _reference_root;

  template <template <class> class Other, bool IsReference>
  constexpr void _init(_iroot &root, std::bool_constant<IsReference>) noexcept
  {
    using model_type =
        _if_t<IsReference, _reference_root<Other, CvValue>, _value_root<Other, value_type>>;
    auto &model  = *::any::_polymorphic_downcast<model_type *>(&root);
    this->value_ = std::addressof(value(model));
  }

  template <template <class> class Other>
  constexpr void _init(_iroot &root, bool is_reference) noexcept
  {
    if (is_reference)
      _init<Other>(root, std::true_type{});
    else
      _init<Other>(root, std::false_type{});
  }

  union
  {
    CvValue *value_ = nullptr;
    _iroot *root_; // points to a _value_root<Extension, value_type>
  };
};

template <template <class> class Interface, class CvValue, template <class> class Extension>
struct _reference_root<Interface, CvValue, iabstract<Extension>>
  : _reference_root<Interface, CvValue>
{
  constexpr explicit _reference_root(_iroot &root, bool is_reference) noexcept
  {
    (*this).template _init<Extension>(root, is_reference);
  }
};

//////////////////////////////////////////////////////////////////////////////////////////
// _reference_proxy_root
template <template <class> class Interface>
struct _reference_proxy_root : iabstract<Interface>
{
  using interface_type                          = iabstract<Interface>;
  static constexpr ::any::_box_kind _box_kind   = ::any::_box_kind::_proxy;
  static constexpr ::any::_root_kind _root_kind = ::any::_root_kind::_reference;

  constexpr _reference_proxy_root() noexcept
  {
    if consteval
    {
      pointer_ = nullptr;
    }
    else
    {
      *::any::start_lifetime_as<_tagged_ptr>(buffer_) = _tagged_ptr();
    }
  }

  _reference_proxy_root(_reference_proxy_root &&)            = delete;
  _reference_proxy_root &operator=(_reference_proxy_root &&) = delete;

  constexpr void _copy(_reference_proxy_root const &other) noexcept
  {
    if consteval
    {
      value(other)._indirect_bind_(*this);
    }
    else
    {
      std::memcpy(buffer_, other.buffer_, sizeof(buffer_));
    }
  }

  constexpr ~_reference_proxy_root()
  {
    if consteval
    {
      _reset_();
    }
  }

  constexpr void swap(_reference_proxy_root &other) noexcept
  {
    if (this != &other)
    {
      if consteval
      {
        std::swap(pointer_, other.pointer_);
      }
      else
      {
        std::swap(buffer_, other.buffer_);
      }
    }
  }

  template <extension_of<Interface> CvModel>
  constexpr void _model_bind_(CvModel &model) noexcept
  {
    static_assert(extension_of<CvModel, Interface>, "CvModel must implement Interface");
    if consteval
    {
      model._indirect_bind_(*this);
    }
    else
    {
      if constexpr (std::derived_from<CvModel, iabstract<Interface>>)
      {
        //! Optimize for when Base derives from iabstract<Interface>. Store the
        //! address of value(other) directly in out as a tagged ptr instead of
        //! introducing an indirection.
        //! @post _is_vptr() == false
        auto &ptr = *::any::start_lifetime_as<_tagged_ptr>(buffer_);
        ptr       = static_cast<iabstract<Interface> *>(std::addressof(::any::_unconst(model)));
      }
      else
      {
        //! @post _is_vptr() == true
        model._indirect_bind_(*this);
      }
    }
  }

  template <class CvModel>
  constexpr void _object_bind_(CvModel &model) noexcept
  {
    static_assert(extension_of<CvModel, Interface>);
    using extension_type        = CvModel::interface_type;
    using element_type          = CvModel::template element_type<CvModel>;
    using model_type            = _reference_model<Interface, element_type, extension_type>;
    constexpr bool is_reference = CvModel::_root_kind == _root_kind::_reference;
    _iroot &root                = const_cast<std::remove_cv_t<CvModel> &>(model);
    ::any::_emplace_into<model_type>(pointer_, buffer_, root, bool(is_reference));
  }

  template <class CvValue>
  constexpr void _value_bind_(CvValue &value) noexcept
  {
    static_assert(!extension_of<CvValue, Interface>);
    using model_type = _reference_model<Interface, CvValue>;
    ::any::_emplace_into<model_type>(pointer_, buffer_, value);
  }

  template <class Self>
  [[nodiscard]]
  constexpr auto &&_value_(this Self &&self) noexcept
  {
    using root_ptr_t      = std::add_pointer_t<_copy_cvref_t<Self, _iroot>>;
    using interface_ref_t = _copy_cvref_t<Self &&, iabstract<Interface>>;
    using interface_ptr_t = std::add_pointer_t<interface_ref_t>;
    if consteval
    {
      return static_cast<interface_ref_t>(
          *::any::_polymorphic_downcast<interface_ptr_t>(self.pointer_));
    }
    else
    {
      ANY_ASSERT(!empty(self));
      auto const ptr       = *::any::start_lifetime_as<_tagged_ptr>(self.buffer_);
      auto *const root_ptr = static_cast<root_ptr_t>(ptr._is_vptr() ? self.buffer_ : ptr._get());
      return static_cast<interface_ref_t>(*::any::_polymorphic_downcast<interface_ptr_t>(root_ptr));
    }
  }

  [[nodiscard]]
  constexpr bool _empty_() const noexcept final override
  {
    if consteval
    {
      return pointer_ == nullptr;
    }
    else
    {
      return *::any::start_lifetime_as<_tagged_ptr>(buffer_) == nullptr;
    }
  }

  constexpr void _reset_() noexcept final override
  {
    if consteval
    {
      delete std::exchange(pointer_, nullptr);
    }
    else
    {
      *::any::start_lifetime_as<_tagged_ptr>(buffer_) = _tagged_ptr();
    }
  }

  [[nodiscard]]
  constexpr type_info const &_type_() const noexcept final override
  {
    return _empty_() ? ANY_TYPEID(void) : type(_value_());
  }

  [[nodiscard]]
  constexpr void *_data_() const noexcept final override
  {
    return _empty_() ? nullptr : data(_value_());
  }

  [[nodiscard]]
  constexpr bool _is_indirect_() const noexcept
  {
    if consteval
    {
      return true;
    }
    else
    {
      return ::any::start_lifetime_as<_tagged_ptr>(buffer_)->_is_vptr();
    }
  }

private:
  union
  {
    _iroot *pointer_ = nullptr; //!< Used in consteval context
    // storage for one vtable ptr and one pointer for the referant
    mutable std::byte buffer_[2 * sizeof(void *)]; //!< Used in runtime context
  };
}; // struct _reference_proxy_root

//////////////////////////////////////////////////////////////////////////////////////////
// bad_any_cast
struct bad_any_cast : std::exception
{
  [[nodiscard]]
  constexpr char const *what() const noexcept override
  {
    return "bad_any_cast";
  }
};

#if __cpp_exceptions
[[noreturn]]
inline void _throw_bad_any_cast()
{
  throw bad_any_cast();
}
#else
[[noreturn]]
inline constexpr void _throw_bad_any_cast() noexcept
{
  ::any::_die("bad_any_cast\n");
}
#endif

//////////////////////////////////////////////////////////////////////////////////////////
//! _any_static_cast
template <class Value>
struct _any_static_cast_t
{
  template <template <class> class Interface, class Base>
  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline constexpr auto *operator()(Interface<Base> *proxy_ptr) const noexcept
  {
    return _cast_<Interface>(proxy_ptr);
  }

  template <template <class> class Interface, class Base>
  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline constexpr auto *operator()(Interface<Base> const *proxy_ptr) const noexcept
  {
    return _cast_<Interface>(proxy_ptr);
  }

private:
  static_assert(_decayed<Value>, "Value must be a decayed type.");

  template <class CvModel>
  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline static constexpr auto *_value_ptr_(CvModel *model) noexcept
  {
    return model != nullptr ? std::addressof(value(*model)) : nullptr;
  }

  template <class CvProxy>
  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline static constexpr bool _is_reference_(CvProxy *proxy_ptr) noexcept
  {
    if constexpr (CvProxy::_root_kind == _root_kind::_reference)
      return (*proxy_ptr)._is_indirect_();
    else
      return false;
  }

  template <template <class> class Interface, class CvProxy>
  [[nodiscard]]
  static constexpr auto *_cast_(CvProxy *proxy_ptr) noexcept
  {
    static_assert(CvProxy::_box_kind == _box_kind::_proxy, "CvProxy must be a proxy type.");
    static_assert(!extension_of<Value, Interface>, "Cannot dynamic cast to an Interface type.");
    using referant_type   = _copy_cvref_t<CvProxy, Value>;

    using value_model     = _copy_cvref_t<CvProxy, _value_root<Interface, Value>>;
    using reference_model = _copy_cvref_t<CvProxy, _reference_root<Interface, referant_type>>;

    // get the address of the model from the proxy:
    auto *model_ptr = std::addressof(value(*proxy_ptr));

    // If CvProxy is a reference proxy that stores the model indirectly, then model_ptr
    // points to a reference model. Otherwise, it points to a value model.
    return _is_reference_(proxy_ptr)
             ? _value_ptr_(::any::_polymorphic_downcast<reference_model *>(model_ptr))
             : _value_ptr_(::any::_polymorphic_downcast<value_model *>(model_ptr));
  }
};

//////////////////////////////////////////////////////////////////////////////////////////
//! _any_static_cast
template <class Value>
struct _any_dynamic_cast_t
{
  template <class CvProxy>
  [[nodiscard]]
  constexpr auto *operator()(CvProxy *proxy_ptr) const noexcept
  {
    return type(*proxy_ptr) == ANY_TYPEID(Value) ? _any_static_cast_t<Value>{}(proxy_ptr) : nullptr;
  }
};

//////////////////////////////////////////////////////////////////////////////////////////
// _basic_cast_t
template <class Value, template <class> class _Cast>
struct _basic_cast_t
{
  template <template <class> class Interface, class Base>
  [[nodiscard]]
  constexpr auto *operator()(Interface<Base> *ptr) const noexcept
  {
    if constexpr (extension_of<Value, Interface>)
      return ptr;
    else if (ptr == nullptr || empty(*ptr))
      return static_cast<Value *>(nullptr);
    else
      return _cast(ptr);
  }

  template <template <class> class Interface, class Base>
  [[nodiscard]]
  constexpr auto *operator()(Interface<Base> const *ptr) const noexcept
  {
    if constexpr (extension_of<Value, Interface>)
      return ptr;
    else if (ptr == nullptr || empty(*ptr))
      return static_cast<Value const *>(nullptr);
    else
      return _cast(ptr);
  }

  template <template <class> class Interface, class Base>
  [[nodiscard]]
  constexpr auto &&operator()(Interface<Base> &&object) const
  {
    auto *ptr = (*this)(std::addressof(object));
    if (ptr == nullptr)
      _throw_bad_any_cast();
    if constexpr (Base::_root_kind == _root_kind::_reference)
      return *ptr;
    else
      return std::move(*ptr);
  }

  template <template <class> class Interface, class Base>
  [[nodiscard]]
  constexpr auto &operator()(Interface<Base> &object) const
  {
    auto *ptr = (*this)(std::addressof(object));
    if (ptr == nullptr)
      _throw_bad_any_cast();
    return *ptr;
  }

  template <template <class> class Interface, class Base>
  [[nodiscard]]
  constexpr auto &operator()(Interface<Base> const &object) const
  {
    auto *ptr = (*this)(std::addressof(object));
    if (ptr == nullptr)
      _throw_bad_any_cast();
    return *ptr;
  }

  template <template <class> class Interface>
  [[nodiscard]]
  constexpr auto *operator()(any_ptr<Interface> const &ptr) const
  {
    return (*this)(ptr.operator->());
  }

  template <template <class> class Interface>
  [[nodiscard]]
  constexpr auto *operator()(any_const_ptr<Interface> const &ptr) const
  {
    return (*this)(ptr.operator->());
  }

private:
  static_assert(_decayed<Value>);
  // The cast is either checked (dynamic) or unchecked (static)
  static constexpr _Cast<Value> _cast{};
};

//////////////////////////////////////////////////////////////////////////////////////////
// any_cast
template <class Value>
struct any_cast_t : _basic_cast_t<Value, _any_dynamic_cast_t>
{
};

template <class Value>
constexpr any_cast_t<Value> any_cast{};

//////////////////////////////////////////////////////////////////////////////////////////
// any_static_cast
template <class Value>
struct any_static_cast_t : _basic_cast_t<Value, _any_static_cast_t>
{
};

template <class Value>
constexpr any_static_cast_t<Value> any_static_cast{};

//////////////////////////////////////////////////////////////////////////////////////////
// imovable
template <class Base>
struct imovable : interface<imovable, Base>
{
  using imovable::interface::interface;

  constexpr virtual void _move_to(_iroot *&, std::span<std::byte>) noexcept
  {
    ::any::_die(_pure_virt_msg, "_move_to");
  }
};

//////////////////////////////////////////////////////////////////////////////////////////
// icopyable
template <class Base>
struct icopyable : interface<icopyable, Base, extends<imovable>>
{
  using icopyable::interface::interface;

  constexpr virtual void _copy_to(_iroot *&, std::span<std::byte>) const
  {
    ::any::_die(_pure_virt_msg, "_copy_to");
  }
};

//////////////////////////////////////////////////////////////////////////////////////////
// utils
template <class Value, template <class> class Interface>
concept _model_of = _decayed<Value> && !std::derived_from<Value, _iroot>;

//////////////////////////////////////////////////////////////////////////////////////////
// any
template <template <class> class Interface>
struct any final : _value_proxy_model<Interface>
{
private:
  template <class Other>
  static constexpr bool _as_large_as =
      iabstract<Interface>::_buffer_size >= Interface<Other>::_buffer_size
      && Other::_root_kind == _root_kind::_value;

public:
  any() = default;

  // Construct from an object that implements the interface (and is not an any<>
  // itself)
  template <_model_of<Interface> Value>
  constexpr any(Value value)
    : any()
  {
    (*this)._emplace_(std::move(value));
  }

  // Implicit derived-to-base conversion constructor
  template <class Other>
    requires extension_of<Interface<Other>, imovable> && (Other::_root_kind == _root_kind::_value)
  constexpr any(Interface<Other> other) noexcept(_as_large_as<Other>)
  {
    (*this)._assign(std::move(other));
  }

  template <class Other>
    requires extension_of<Interface<Other>, icopyable>
          && (Other::_root_kind == _root_kind::_reference)
  constexpr any(Interface<Other> const &other)
  {
    Interface<Other> temp;
    temp._copy(other);
    (*this)._assign(std::move(temp));
  }

  template <_model_of<Interface> Value>
  constexpr any &operator=(Value value)
  {
    reset(*this);
    (*this)._emplace_(std::move(value));
    return *this;
  }

  // Implicit derived-to-base conversion constructor
  template <class Other>
    requires extension_of<Interface<Other>, imovable> && (Other::_root_kind == _root_kind::_value)
  constexpr any &operator=(Interface<Other> other) noexcept(_as_large_as<Other>)
  {
    reset(*this);
    (*this)._assign(std::move(other));
    return *this;
  }

  template <class Other>
    requires extension_of<Interface<Other>, icopyable>
          && (Other::_root_kind == _root_kind::_reference)
  constexpr any &operator=(Interface<Other> const &other)
  {
    // Guard against self-assignment when other is a reference to *this
    if (data(other) == data(*this))
      return *this;

    Interface<Other> temp;
    temp._copy(other);

    reset(*this);
    (*this)._assign(std::move(temp));
    return *this;
  }

  friend constexpr void swap(any &lhs, any &rhs) noexcept
    requires any::_movable
  {
    lhs.swap(rhs);
  }

private:
  // Assigning from a type that extends Interface. Its buffer may be larger than
  // ours, or it may be a reference type, so we can be only conditionally
  // noexcept.
  template <class Other>
    requires extension_of<Interface<Other>, imovable>
  constexpr void _assign(Interface<Other> &&other) noexcept(_as_large_as<Other>)
  {
    constexpr bool ptr_convertible = std::derived_from<Other, iabstract<Interface>>;

    if (empty(other))
    {
      return;
    }
    else if constexpr (Other::_root_kind == _root_kind::_reference || !ptr_convertible)
    {
      return other._slice_to_(*this);
    }
    else if (other._in_situ_())
    {
      return other._slice_to_(*this);
    }
    else if consteval
    {
      (*this).pointer_ = std::exchange(other.pointer_, nullptr);
    }
    else
    {
      auto &ptr = *::any::start_lifetime_as<_tagged_ptr>((*this).buffer_);
      ptr       = *::any::start_lifetime_as<_tagged_ptr>(other.buffer_);
    }
  }

  static_assert(sizeof(iabstract<Interface>) == sizeof(void *)); // sanity check
};

//////////////////////////////////////////////////////////////////////////////////////////
// _any_ptr_base
template <template <class> class Interface>
struct _any_ptr_base
{
  _any_ptr_base() = default;

  constexpr _any_ptr_base(std::nullptr_t) noexcept
    : reference_()
  {
  }

  constexpr _any_ptr_base(_any_ptr_base const &other) noexcept
    : reference_()
  {
    (*this)._proxy_assign(std::addressof(other.reference_));
  }

  template <template <class> class OtherInterface>
    requires extension_of<iabstract<OtherInterface>, Interface>
  constexpr _any_ptr_base(_any_ptr_base<OtherInterface> const &other) noexcept
    : reference_()
  {
    (*this)._proxy_assign(std::addressof(other.reference_));
  }

  constexpr _any_ptr_base &operator=(_any_ptr_base const &other) noexcept
  {
    reset(reference_);
    (*this)._proxy_assign(std::addressof(other.reference_));
    return *this;
  }

  constexpr _any_ptr_base &operator=(std::nullptr_t) noexcept
  {
    reset(reference_);
    return *this;
  }

  template <template <class> class OtherInterface>
    requires extension_of<iabstract<OtherInterface>, Interface>
  constexpr _any_ptr_base &operator=(_any_ptr_base<OtherInterface> const &other) noexcept
  {
    reset(reference_);
    (*this)._proxy_assign(std::addressof(other.reference_));
    return *this;
  }

  friend constexpr void swap(_any_ptr_base &lhs, _any_ptr_base &rhs) noexcept
  {
    lhs.reference_.swap(rhs.reference_);
  }

  [[nodiscard]]
  constexpr bool operator==(_any_ptr_base const &other) const noexcept
  {
    return data(reference_) == data(other.reference_);
  }

private:
  static_assert(sizeof(iabstract<Interface>) == sizeof(void *)); // sanity check

  template <template <class> class>
  friend struct _any_ptr_base;

  friend struct any_ptr<Interface>;
  friend struct any_const_ptr<Interface>;

  //! @param other A pointer to a value proxy model implementing Interface.
  template <extension_of<Interface> CvValueProxy>
  constexpr void _proxy_assign(CvValueProxy *proxy_ptr) noexcept
  {
    static_assert(CvValueProxy::_box_kind == _box_kind::_proxy);
    constexpr bool is_const = std::is_const_v<CvValueProxy>;

    if (proxy_ptr == nullptr || empty(*proxy_ptr))
      return;
    // Optimize for when CvValueProxy derives from iabstract<Interface>. Store the address
    // of value(other) directly in out as a tagged ptr instead of introducing an
    // indirection.
    else if constexpr (std::derived_from<CvValueProxy, iabstract<Interface>>)
      reference_._model_bind_(::any::_as_const_if<is_const>(value(*proxy_ptr)));
    else
      value(*proxy_ptr)._indirect_bind_(reference_);
  }

  //! @param other A pointer to a reference proxy model implementing Interface.
  template <extension_of<Interface> CvReferenceProxy>
    requires(CvReferenceProxy::_root_kind == _root_kind::_reference)
  constexpr void _proxy_assign(CvReferenceProxy *proxy_ptr) noexcept
  {
    static_assert(CvReferenceProxy::_box_kind == _box_kind::_proxy);
    using model_type        = _reference_proxy_model<Interface>;
    constexpr bool is_const = std::is_const_v<CvReferenceProxy>;

    if (proxy_ptr == nullptr || empty(*proxy_ptr))
      return;
    // in the case where CvReferenceProxy is a base class of model_type, we can simply
    // downcast and copy the model directly.
    else if constexpr (std::derived_from<model_type, CvReferenceProxy>)
      reference_._copy(*::any::_polymorphic_downcast<model_type const *>(proxy_ptr));
    // Otherwise, we are assigning from a derived reference to a base reference, and the
    // other reference is indirect (i.e., it holds a _reference_model in its buffer). We
    // need to copy the referant model.
    else if ((*proxy_ptr)._is_indirect_())
      value(*proxy_ptr)._indirect_bind_(reference_);
    else
      reference_._model_bind_(::any::_as_const_if<is_const>(value(*proxy_ptr)));
  }

  template <class CvValue>
  constexpr void _value_assign(CvValue *value_ptr) noexcept
  {
    if (value_ptr != nullptr)
      reference_._value_bind_(*value_ptr);
  }

  // the proxy model is mutable so that a const any_ptr can return non-const
  // references from operator-> and operator*.
  mutable _reference_proxy_model<Interface> reference_;
};

//////////////////////////////////////////////////////////////////////////////////////////
// any_ptr
template <template <class> class Interface>
struct any_ptr : _any_ptr_base<Interface>
{
  using _any_ptr_base<Interface>::_any_ptr_base;
  using _any_ptr_base<Interface>::operator=;

  // Disable const-to-mutable conversions:
  template <template <class> class Other>
  any_ptr(any_const_ptr<Other> const &) = delete;
  template <template <class> class Other>
  any_ptr &operator=(any_const_ptr<Other> const &) = delete;

  template <_model_of<Interface> Value>
  constexpr any_ptr(Value *value_ptr) noexcept
    : _any_ptr_base<Interface>()
  {
    (*this)._value_assign(value_ptr);
  }

  template <extension_of<Interface> Proxy>
  constexpr any_ptr(Proxy *proxy_ptr) noexcept
    : _any_ptr_base<Interface>()
  {
    (*this)._proxy_assign(proxy_ptr);
  }

  template <extension_of<Interface> Proxy>
  any_ptr(Proxy const *) = delete;

  template <_model_of<Interface> Value>
  constexpr any_ptr &operator=(Value *value_ptr) noexcept
  {
    reset((*this).reference_);
    (*this)._value_assign(value_ptr);
    return *this;
  }

  template <extension_of<Interface> Proxy>
  constexpr any_ptr &operator=(Proxy *proxy_ptr) noexcept
  {
    reset((*this).reference_);
    (*this)._proxy_assign(proxy_ptr);
    return *this;
  }

  template <extension_of<Interface> Proxy>
  any_ptr &operator=(Proxy const *proxy_ptr) = delete;

  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline constexpr auto *operator->() const noexcept
  {
    return std::addressof((*this).reference_);
  }

  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline constexpr auto &operator*() const noexcept
  {
    return (*this).reference_;
  }
};

template <template <class> class Interface, class Base>
any_ptr(Interface<Base> *) -> any_ptr<Interface>;

//////////////////////////////////////////////////////////////////////////////////////////
// any_const_ptr
template <template <class> class Interface>
struct any_const_ptr : _any_ptr_base<Interface>
{
  using _any_ptr_base<Interface>::_any_ptr_base;
  using _any_ptr_base<Interface>::operator=;

  template <_model_of<Interface> Value>
  constexpr any_const_ptr(Value const *value_ptr) noexcept
    : _any_ptr_base<Interface>()
  {
    (*this)._value_assign(value_ptr);
  }

  template <extension_of<Interface> Proxy>
  constexpr any_const_ptr(Proxy const *proxy_ptr) noexcept
    : _any_ptr_base<Interface>()
  {
    (*this)._proxy_assign(proxy_ptr);
  }

  template <_model_of<Interface> Value>
  constexpr any_const_ptr &operator=(Value const *value_ptr) noexcept
  {
    reset((*this).reference_);
    (*this)._value_assign(value_ptr);
    return *this;
  }

  template <extension_of<Interface> Proxy>
  constexpr any_const_ptr &operator=(Proxy const *proxy_ptr) noexcept
  {
    reset((*this).reference_);
    (*this)._proxy_assign(proxy_ptr);
    return *this;
  }

  friend constexpr void swap(any_const_ptr &a, any_const_ptr &b) noexcept
  {
    a.reference_.swap(b.reference_);
  }

  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline constexpr auto const *operator->() const noexcept
  {
    return std::addressof((*this).reference_);
  }

  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline constexpr auto const &operator*() const noexcept
  {
    return (*this).reference_;
  }
};

template <template <class> class Interface, class Base>
any_const_ptr(Interface<Base> const *) -> any_const_ptr<Interface>;

//////////////////////////////////////////////////////////////////////////////////////////
// iequality_comparable
template <class Base>
struct iequality_comparable : interface<iequality_comparable, Base>
{
  using iequality_comparable::interface::interface;

  template <class Other>
  [[nodiscard]]
  constexpr bool operator==(iequality_comparable<Other> const &other) const
  {
    return _equal_to(::any::caddressof(other));
  }

private:
  [[nodiscard]]
  constexpr virtual bool _equal_to(any_const_ptr<iequality_comparable> other) const
  {
    auto const &type = ::any::type(*this);

    if (type != ::any::type(*other))
      return false;

    if (type == ANY_TYPEID(void))
      return true;

    using value_type = value_of_t<iequality_comparable>;
    return value(*this) == ::any::any_static_cast<value_type>(*other);
  }
};

//////////////////////////////////////////////////////////////////////////////////////////
// isemiregular
template <class Base>
struct isemiregular : interface<isemiregular, Base, extends<icopyable, iequality_comparable>>
{
  using isemiregular::interface::interface;
};

} // namespace any

ANY_DIAG_POP
