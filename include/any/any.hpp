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
//////////////////////////////////////////////////////////////////////////////////////////
// forward declarations

// any types
template <template <class> class Interface>
struct any;

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
using _bases_of = Interface<_iroot>::bases_type;

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
[[maybe_unused]] constexpr struct _value_t
{
  template <class T>
  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline constexpr auto &operator()(T &t) const noexcept
  {
    return t._value();
  }
} value{};

[[maybe_unused]] constexpr struct _empty_t
{
  template <class T>
  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline constexpr bool operator()(T const &t) const noexcept
  {
    return t._empty();
  }
} empty{};

[[maybe_unused]] constexpr struct _reset_t
{
  template <class T>
  [[ANY_ALWAYS_INLINE]]
  inline constexpr void operator()(T &t) const noexcept
  {
    t._reset();
  }
} reset{};

[[maybe_unused]] constexpr struct _type_t
{
  template <class T>
  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline constexpr type_info const &operator()(T const &t) const noexcept
  {
    return t._type();
  }
} type{};

[[maybe_unused]] constexpr struct _data
{
  template <class T>
  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline constexpr auto *operator()(T &t) const noexcept
  {
    return t._data();
  }
} data{};

[[maybe_unused]] constexpr struct caddressof_t
{
  template <template <class> class Interface, class Base>
  [[nodiscard]]
  constexpr auto operator()(Interface<Base> const &iface) const noexcept
  {
    return any_const_ptr<Interface>(std::addressof(iface));
  }
} caddressof{};

[[maybe_unused]] constexpr struct addressof_t : caddressof_t
{
  using caddressof_t::operator();

  template <template <class> class Interface, class Base>
  [[nodiscard]]
  constexpr auto operator()(Interface<Base> &iface) const noexcept
  {
    return any_ptr<Interface>(std::addressof(iface));
  }
} addressof{};

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

//////////////////////////////////////////////////////////////////////////////////////////
// _emplace_into
template <class Model, class... Args>
constexpr Model &_emplace_into([[maybe_unused]] _iroot *&pointer,
                               [[maybe_unused]] std::span<std::byte> buffer,
                               Args &&...args)
{
  static_assert(_decayed<Model>);
  if ANY_CONSTEVAL
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

// template <class Root>
// concept root = requires (Root& root)
// {
//   root._value();
//   root._reset();
//   { root._type() } -> std::same_as<const type_info &>;
//   { root._data() } -> std::same_as<void *>;
//   { root._empty() } -> std::same_as<bool>;
// };

//! @c iabstract must be an alias in order for @c iabstract<Derived> to be
//! derived from
//! @c iabstract<Base>. @c iabstract<Derived> is an alias for @c
//! Derived<Base<_iroot>>.
template <template <class> class Interface, class BaseInterfaces = _bases_of<Interface>>
using iabstract = Interface<_mcall<BaseInterfaces, _iroot>>;

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
struct _value_proxy_model : Interface<_mcall<_bases_of<Interface>, _value_proxy_root<Interface>>>
{
};

// reference
template <template <class> class Interface, class Value>
struct _reference_root;

template <template <class> class Interface, class Value>
struct _reference_model : Interface<_mcall<_bases_of<Interface>, _reference_root<Interface, Value>>>
{
  using _base_t = Interface<_mcall<_bases_of<Interface>, _reference_root<Interface, Value>>>;
  using _base_t::_base_t;
};

// reference proxy
template <template <class> class Interface>
struct _reference_proxy_root;

template <template <class> class Interface>
struct _reference_proxy_model
  : Interface<_mcall<_bases_of<Interface>, _reference_proxy_root<Interface>>>
{
};

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

constexpr size_t default_buffer_size = 3 * sizeof(void *);

template <class Interface, _box_kind BoxKind>
concept _has_box_kind = std::remove_reference_t<Interface>::_box_kind == BoxKind;

// Without the check against _has_box_kind, this concept would always be
// satisfied when building an object model or a proxy model because of the
// abstract implementation of BaseInterface in the iabstract layer.
//
// any<Derived>
//   : _value_proxy_model<Derived, V>
//       : Derived<Base<_value_proxy_root<Derived, V>>>    // box_kind == object
//         ^^^^^^^        : Derived<Base<_iroot>>         // box_kind ==
//         abstract
//                          ^^^^^^^
template <class Interface, template <class> class BaseInterface>
concept _already_implements = requires(Interface const &iface) {
  { ::any::interface_cast<BaseInterface>(iface) } -> _has_box_kind<Interface::_box_kind>;
};

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

constexpr char const *_pure_virt_msg = "internal error: pure virtual %s() called\n";

// If we are slicing into a buffer that is smaller than our own, then slicing
// may throw.
template <class Interface, class Base, size_t BufferSize>
concept _nothrow_slice = (Base::_root_kind == _root_kind::_value)
                      && (Base::_box_kind != _box_kind::_abstract)
                      && (Interface::buffer_size >= BufferSize);

//////////////////////////////////////////////////////////////////////////////////////////
//! interface
template <template <class> class Interface,
          class Base,
          class BaseInterfaces = extends<>,
          size_t BufferSize    = default_buffer_size>
struct interface : Base
{
  using bases_type      = BaseInterfaces;
  using _interface_type = iabstract<Interface, BaseInterfaces>;
  using Base::_indirect_bind;
  using Base::_slice_to;
  using Base::Base;

  static constexpr size_t buffer_size =
      BufferSize > Base::buffer_size ? BufferSize : Base::buffer_size;

  static constexpr bool _nothrow_slice = ::any::_nothrow_slice<_interface_type, Base, buffer_size>;

  constexpr ~interface()               = default;

  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline constexpr auto &_value() noexcept
  {
    if constexpr (Base::_box_kind == _box_kind::_abstract)
      return ::any::_die<_interface_type &>(_pure_virt_msg, "value");
    else
      return Base::_value();
  }

  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline constexpr auto const &_value() const noexcept
  {
    if constexpr (Base::_box_kind == _box_kind::_abstract)
      return ::any::_die<_interface_type const &>(_pure_virt_msg, "value");
    else
      return Base::_value();
  }

  //! @pre Base::_box_kind != _box_kind::_proxy || !empty(*this)
  constexpr virtual void _slice_to(_value_proxy_root<Interface> &out) noexcept(_nothrow_slice)
  {
    if constexpr (Base::_box_kind == _box_kind::_abstract)
    {
      ::any::_die(_pure_virt_msg, "slice");
    }
    else if constexpr (Base::_box_kind == _box_kind::_object)
    {
      out.emplace(std::move(value(*this))); // potentially throwing
    }
    else
    {
      value(*this)._slice_to(out);
      reset(*this);
    }
  }

  [[ANY_ALWAYS_INLINE]]
  inline constexpr void _indirect_bind(_reference_proxy_root<Interface> &out) noexcept
  {
    return std::as_const(*this)._indirect_bind(out, false);
  }

  constexpr virtual void _indirect_bind(_reference_proxy_root<Interface> &out,
                                        bool is_const = true) const noexcept
  {
    if constexpr (Base::_box_kind == _box_kind::_abstract)
    {
      ::any::_die(_pure_virt_msg, "bind");
    }
    else if constexpr (Base::_box_kind == _box_kind::_object)
    {
      if (is_const)
        out._direct_bind(value(*this));
      else
        out._direct_bind(value(::any::_unconst(*this)));
    }
    else
    {
      ANY_ASSERT(!empty(*this));
      if (is_const)
        value(*this)._indirect_bind(out, true);
      else
        value(::any::_unconst(*this))._indirect_bind(out, false);
    }
  }
};

//////////////////////////////////////////////////////////////////////////////////////////
// _iroot
struct _iroot
{
  static constexpr ::any::_box_kind _box_kind   = ::any::_box_kind::_abstract;
  static constexpr ::any::_root_kind _root_kind = ::any::_root_kind::_value;
  static constexpr size_t buffer_size           = sizeof(_tagged_ptr); // minimum size
  using bases_type                              = extends<>;

  // needed by MSVC for EBO to work for some reason:
  constexpr virtual ~_iroot() = default;

  [[nodiscard]]
  constexpr virtual bool _empty() const noexcept
  {
    return ::any::_die<bool>(_pure_virt_msg, "empty");
  }

  constexpr virtual void _reset() noexcept
  {
    ::any::_die(_pure_virt_msg, "reset");
  }

  [[nodiscard]]
  constexpr virtual type_info const &_type() const noexcept
  {
    return ::any::_die<type_info const &>(_pure_virt_msg, "type");
  }

  [[nodiscard]]
  constexpr virtual void *_data() const noexcept
  {
    return ::any::_die<void *>(_pure_virt_msg, "data");
  }

  void _slice_to() noexcept               = delete;
  void _indirect_bind() const noexcept = delete;
};

//////////////////////////////////////////////////////////////////////////////////////////
// _value_root
template <template <class> class Interface, class Value>
struct _value_root : iabstract<Interface>
{
  using interface_type                        = iabstract<Interface>;
  static constexpr ::any::_box_kind _box_kind = ::any::_box_kind::_object;

  constexpr explicit _value_root(Value val) noexcept
    : value(std::move(val))
  {
  }

  [[nodiscard]]
  constexpr Value &_value() noexcept
  {
    return value;
  }

  [[nodiscard]]
  constexpr Value const &_value() const noexcept
  {
    return value;
  }

  [[nodiscard]]
  constexpr bool _empty() const noexcept final override
  {
    return false;
  }

  constexpr void _reset() noexcept final override
  {
    // no-op
  }

  [[nodiscard]]
  constexpr type_info const &_type() const noexcept final override
  {
    return TYPEID(Value);
  }

  [[nodiscard]]
  constexpr void *_data() const noexcept final override
  {
    return const_cast<void *>(static_cast<void const *>(std::addressof(_value())));
  }

  Value value;
};

// A specialization of _value_root to take advantage of EBO (empty base
// optimization):
template <template <class> class Interface, class Value>
  requires std::is_empty_v<Value> && (!std::is_final_v<Value>)
struct [[ANY_EMPTY_BASES]] _value_root<Interface, Value>
  : iabstract<Interface>
  , private Value
{
  using interface_type                        = iabstract<Interface>;
  static constexpr ::any::_box_kind _box_kind = ::any::_box_kind::_object;

  constexpr explicit _value_root(Value val) noexcept
    : Value(std::move(val))
  {
  }

  [[nodiscard]]
  constexpr Value &_value() noexcept
  {
    return *this;
  }

  [[nodiscard]]
  constexpr Value const &_value() const noexcept
  {
    return *this;
  }

  [[nodiscard]]
  constexpr bool _empty() const noexcept final override
  {
    return false;
  }

  constexpr void _reset() noexcept final override
  {
    if ANY_CONSTEVAL
    {
      delete this;
    }
  }

  [[nodiscard]]
  constexpr type_info const &_type() const noexcept final override
  {
    return TYPEID(Value);
  }

  [[nodiscard]]
  constexpr void *_data() const noexcept final override
  {
    return const_cast<void *>(static_cast<void const *>(std::addressof(_value())));
  }
};

//////////////////////////////////////////////////////////////////////////////////////////
// _value_proxy_root
template <template <class> class Interface>
struct _value_proxy_root : iabstract<Interface>
{
  using interface_type                        = iabstract<Interface>;
  static constexpr ::any::_box_kind _box_kind = ::any::_box_kind::_proxy;

  static constexpr bool _movable              = extension_of<iabstract<Interface>, imovable>;
  static constexpr bool _copyable             = extension_of<iabstract<Interface>, icopyable>;

  [[ANY_ALWAYS_INLINE]]
  inline constexpr _value_proxy_root() noexcept
  {
    if ANY_CONSTEVAL
    {
      pointer = nullptr;
    }
    else
    {
      *::any::start_lifetime_as<_tagged_ptr>(buffer) = _tagged_ptr();
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
      value(other)._copy_to(pointer, buffer);
  }

  constexpr ~_value_proxy_root()
  {
    _reset();
  }

  constexpr _value_proxy_root &operator=(_value_proxy_root &&other) noexcept
    requires _movable
  {
    if (this != &other)
    {
      _reset();
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
    if ANY_CONSTEVAL
    {
      std::swap(pointer, other.pointer);
    }
    else
    {
      if (this == &other)
        return;

      auto &this_ptr = *::any::start_lifetime_as<_tagged_ptr>(buffer);
      auto &that_ptr = *::any::start_lifetime_as<_tagged_ptr>(other.buffer);

      // This also covers the case where both this_ptr and that_ptr are null.
      if (!this_ptr._is_vptr() && !that_ptr._is_vptr())
        return std::swap(this_ptr, that_ptr);

      if (this_ptr == nullptr)
        return value(other)._move_to(pointer, buffer);

      if (that_ptr == nullptr)
        return value(*this)._move_to(other.pointer, other.buffer);

      auto temp = std::move(*this);
      value(other)._move_to(pointer, buffer);
      value(temp)._move_to(other.pointer, other.buffer);
    }
  }

  template <class Value, class... Args>
  constexpr Value &_emplace(Args &&...args)
  {
    static_assert(_decayed<Value>, "Value must be an object type.");
    using model_type = _value_model<Interface, Value>;
    auto &model = ::any::_emplace_into<model_type>(pointer, buffer, std::forward<Args>(args)...);
    return model._value();
  }

  template <int = 0, class CvRefValue, class Value = std::decay_t<CvRefValue>>
  constexpr Value &_emplace(CvRefValue &&value)
  {
    return _emplace<Value>(std::forward<CvRefValue>(value));
  }

  template <class Value, class... Args>
  constexpr Value &emplace(Args &&...args)
  {
    _reset();
    return _emplace<Value>(std::forward<Args>(args)...);
  }

  template <int = 0, class CvRefValue, class Value = std::decay_t<CvRefValue>>
  constexpr Value &emplace(CvRefValue &&value)
  {
    _reset();
    return _emplace<Value>(std::forward<CvRefValue>(value));
  }

  [[nodiscard]]
  constexpr iabstract<Interface> &_value() noexcept
  {
    if ANY_CONSTEVAL
    {
      return *::any::_polymorphic_downcast<iabstract<Interface> *>(pointer);
    }
    else
    {
      auto const ptr = *::any::start_lifetime_as<_tagged_ptr>(buffer);
      ANY_ASSERT(ptr != nullptr);
      auto *root_ptr = static_cast<_iroot *>(ptr._is_vptr() ? buffer : ptr._get());
      return *::any::_polymorphic_downcast<iabstract<Interface> *>(root_ptr);
    }
  }

  [[nodiscard]]
  constexpr iabstract<Interface> const &_value() const noexcept
  {
    if ANY_CONSTEVAL
    {
      return *::any::_polymorphic_downcast<iabstract<Interface> const *>(pointer);
    }
    else
    {
      auto const ptr = *::any::start_lifetime_as<_tagged_ptr>(buffer);
      ANY_ASSERT(ptr != nullptr);
      auto *root_ptr = static_cast<_iroot const *>(ptr._is_vptr() ? buffer : ptr._get());
      return *::any::_polymorphic_downcast<iabstract<Interface> const *>(root_ptr);
    }
  }

  [[nodiscard]]
  constexpr bool _empty() const noexcept final override
  {
    if ANY_CONSTEVAL
    {
      return pointer == nullptr;
    }
    else
    {
      return *::any::start_lifetime_as<_tagged_ptr>(buffer) == nullptr;
    }
  }

  [[ANY_ALWAYS_INLINE]]
  inline constexpr void _reset() noexcept final override
  {
    if ANY_CONSTEVAL
    {
      delete std::exchange(pointer, nullptr);
    }
    else
    {
      auto &ptr = *::any::start_lifetime_as<_tagged_ptr>(buffer);
      if (ptr == nullptr)
        return;
      else if (ptr._is_vptr())
        std::destroy_at(std::addressof(_value()));
      else
        delete std::addressof(_value());

      ptr = _tagged_ptr();
    }
  }

  [[nodiscard]]
  constexpr type_info const &_type() const noexcept final override
  {
    return _empty() ? TYPEID(void) : type(_value());
  }

  [[nodiscard]]
  constexpr void *_data() const noexcept final override
  {
    return _empty() ? nullptr : data(_value());
  }

  [[nodiscard]]
  constexpr bool _in_situ() const noexcept
  {
    if ANY_CONSTEVAL
    {
      return false;
    }
    else
    {
      return ::any::start_lifetime_as<_tagged_ptr>(buffer)->_is_vptr();
    }
  }

  union
  {
    _iroot *pointer = nullptr;                           //!< Used in consteval context
    std::byte buffer[iabstract<Interface>::buffer_size]; //!< Used in runtime context
  };
};

//////////////////////////////////////////////////////////////////////////////////////////
// _reference_root
template <template <class> class Interface, class CvValue>
struct _reference_root : iabstract<Interface>
{
  static_assert(!extension_of<CvValue, Interface>,
                "Value must be a concrete type, not an Interface type.");
  using interface_type                          = iabstract<Interface>;
  static constexpr ::any::_box_kind _box_kind   = ::any::_box_kind::_object;
  static constexpr ::any::_root_kind _root_kind = ::any::_root_kind::_reference;

  constexpr explicit _reference_root(CvValue &value) noexcept
    : value_(std::addressof(value))
  {
  }

  [[nodiscard]]
  constexpr auto &_value() noexcept
  {
    return ::any::_unconst(*value_);
  }

  [[nodiscard]]
  constexpr auto &_value() const noexcept
  {
    return *value_;
  }

  [[nodiscard]]
  constexpr bool _empty() const noexcept final override
  {
    return false;
  }

  constexpr void _reset() noexcept final override
  {
    // no-op
  }

  [[nodiscard]]
  constexpr type_info const &_type() const noexcept final override
  {
    return TYPEID(CvValue);
  }

  [[nodiscard]]
  constexpr void *_data() const noexcept final override
  {
    return const_cast<void *>(static_cast<void const *>(value_));
  }

private:
  CvValue *value_;
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
    if ANY_CONSTEVAL
    {
      pointer = nullptr;
    }
    else
    {
      *::any::start_lifetime_as<_tagged_ptr>(buffer) = _tagged_ptr();
    }
  }

  constexpr _reference_proxy_root(_reference_proxy_root &&__other) noexcept
    : _reference_proxy_root()
  {
    swap(__other);
  }

  constexpr _reference_proxy_root(_reference_proxy_root const &__other) noexcept
  {
    if ANY_CONSTEVAL
    {
      value(__other)._indirect_bind(*this);
    }
    else
    {
      std::memcpy(buffer, __other.buffer, sizeof(buffer));
    }
  }

  constexpr _reference_proxy_root &operator=(_reference_proxy_root &&__other) noexcept
  {
    if (this != &__other)
    {
      _reset();
      swap(__other);
    }
    return *this;
  }

  constexpr _reference_proxy_root &operator=(_reference_proxy_root const &__other) noexcept
  {
    if (this != &__other)
    {
      _reset();
      _reference_proxy_root(__other).swap(*this);
    }
    return *this;
  }

  constexpr ~_reference_proxy_root()
  {
    if ANY_CONSTEVAL
    {
      _reset();
    }
  }

  constexpr void swap(_reference_proxy_root &other) noexcept
  {
    if (this != &other)
    {
      if ANY_CONSTEVAL
      {
        std::swap(pointer, other.pointer);
      }
      else
      {
        std::swap(buffer, other.buffer);
      }
    }
  }

  template <class CvProxy>
  constexpr void _proxy_bind(CvProxy &proxy) noexcept
  {
    static_assert(extension_of<CvProxy, Interface>, "CvProxy must implement Interface");
    //! @c other should refer to a value model
    static_assert(CvProxy::_root_kind == _root_kind::_value,
                  "CvDerived should not be a reference model");
    if ANY_CONSTEVAL
    {
      proxy._indirect_bind(*this);
    }
    else
    {
      if (std::derived_from<CvProxy, iabstract<Interface>>)
      {
        //! Optimize for when Base derives from iabstract<Interface>. Store the
        //! address of value(other) directly in out as a tagged ptr instead of
        //! introducing an indirection.
        //! @post _is_vptr() == false
        auto &ptr = *::any::start_lifetime_as<_tagged_ptr>(buffer);
        ptr       = static_cast<iabstract<Interface> *>(std::addressof(::any::_unconst(proxy)));
      }
      else
      {
        //! @post _is_vptr() == true
        proxy._indirect_bind(*this);
      }
    }
  }

  template <class CvValue>
  constexpr void _direct_bind(CvValue &value) noexcept
  {
    static_assert(!extension_of<CvValue, Interface>);
    using model_type = _reference_model<Interface, CvValue>;
    ::any::_emplace_into<model_type>(pointer, buffer, value);
  }

  [[nodiscard]]
  constexpr iabstract<Interface> &_value() noexcept
  {
    if ANY_CONSTEVAL
    {
      return *::any::_polymorphic_downcast<iabstract<Interface> *>(pointer);
    }
    else
    {
      ANY_ASSERT(!_empty());
      auto const ptr       = *::any::start_lifetime_as<_tagged_ptr>(buffer);
      auto *const root_ptr = static_cast<_iroot *>(ptr._is_vptr() ? buffer : ptr._get());
      return *::any::_polymorphic_downcast<iabstract<Interface> *>(root_ptr);
    }
  }

  [[nodiscard]]
  constexpr iabstract<Interface> const &_value() const noexcept
  {
    if ANY_CONSTEVAL
    {
      return *::any::_polymorphic_downcast<iabstract<Interface> const *>(pointer);
    }
    else
    {
      ANY_ASSERT(!_empty());
      auto const ptr       = *::any::start_lifetime_as<_tagged_ptr>(buffer);
      auto *const root_ptr = static_cast<_iroot const *>(ptr._is_vptr() ? buffer : ptr._get());
      return *::any::_polymorphic_downcast<iabstract<Interface> const *>(root_ptr);
    }
  }

  [[nodiscard]]
  constexpr bool _empty() const noexcept final override
  {
    if ANY_CONSTEVAL
    {
      return pointer == nullptr;
    }
    else
    {
      return *::any::start_lifetime_as<_tagged_ptr>(buffer) == nullptr;
    }
  }

  constexpr void _reset() noexcept final override
  {
    if ANY_CONSTEVAL
    {
      delete std::exchange(pointer, nullptr);
    }
    else
    {
      *::any::start_lifetime_as<_tagged_ptr>(buffer) = _tagged_ptr();
    }
  }

  [[nodiscard]]
  constexpr type_info const &_type() const noexcept final override
  {
    return _empty() ? TYPEID(void) : type(_value());
  }

  [[nodiscard]]
  constexpr void *_data() const noexcept final override
  {
    return _empty() ? nullptr : data(_value());
  }

  [[nodiscard]]
  constexpr bool _is_indirect() const noexcept
  {
    if ANY_CONSTEVAL
    {
      return true;
    }
    else
    {
      return ::any::start_lifetime_as<_tagged_ptr>(buffer)->_is_vptr();
    }
  }

private:
  union
  {
    _iroot *pointer = nullptr; //!< Used in consteval context
    // storage for one vtable ptr and one pointer for the referant
    mutable std::byte buffer[2 * sizeof(void *)]; //!< Used in runtime context
  };
};

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
    return _cast<Interface>(proxy_ptr);
  }

  template <template <class> class Interface, class Base>
  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline constexpr auto *operator()(Interface<Base> const *proxy_ptr) const noexcept
  {
    return _cast<Interface>(proxy_ptr);
  }

private:
  static_assert(_decayed<Value>, "Value must be a decayed type.");

  template <class CvModel>
  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline static constexpr auto *_value_ptr(CvModel *model) noexcept
  {
    return model != nullptr ? std::addressof(value(*model)) : nullptr;
  }

  template <class CvProxy>
  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline static constexpr bool _is_reference(CvProxy *proxy_ptr) noexcept
  {
    if constexpr (CvProxy::_root_kind == _root_kind::_reference)
      return (*proxy_ptr)._is_indirect();
    else
      return false;
  }

  template <template <class> class Interface, class CvProxy>
  [[nodiscard]]
  static constexpr auto *_cast(CvProxy *proxy_ptr) noexcept
  {
    static_assert(CvProxy::_box_kind == _box_kind::_proxy, "CvProxy must be a proxy type.");
    static_assert(!extension_of<Value, Interface>, "Cannot dynamic cast to an Interface type.");
    constexpr bool is_const = std::is_const_v<CvProxy>;
    using value_model       = _const_if<is_const, _value_root<Interface, Value>>;
    using referant_type     = _const_if<is_const, Value>;
    using reference_model   = _const_if<is_const, _reference_root<Interface, referant_type>>;

    // get the address of the model from the proxy:
    auto *model_ptr = std::addressof(value(*proxy_ptr));

    // If CvProxy is a reference proxy that stores the model indirectly, then model_ptr
    // points to a reference_model. Otherwise, it points to a value_model.
    return _is_reference(proxy_ptr)
             ? _value_ptr(::any::_polymorphic_downcast<reference_model *>(model_ptr))
             : _value_ptr(::any::_polymorphic_downcast<value_model *>(model_ptr));
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
    return type(*proxy_ptr) == TYPEID(Value) ? _any_static_cast_t<Value>{}(proxy_ptr) : nullptr;
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
      iabstract<Interface>::buffer_size >= Interface<Other>::buffer_size
      && Other::_root_kind == _root_kind::_value;

public:
  any() = default;

  // Construct from an object that implements the interface (and is not an any<>
  // itself)
  template <_model_of<Interface> Value>
  constexpr any(Value value)
    : any()
  {
    (*this)._emplace(std::move(value));
  }

  // Implicit derived-to-base conversion constructor
  template <class Other>
    requires extension_of<Interface<Other>, imovable>
  constexpr any(Interface<Other> other) noexcept(_as_large_as<Other>)
  {
    (*this)._assign(std::move(other));
  }

  template <_model_of<Interface> Value>
  constexpr any &operator=(Value value)
  {
    reset(*this);
    (*this)._emplace(std::move(value));
    return *this;
  }

  // Implicit derived-to-base conversion constructor
  template <class Other>
    requires extension_of<Interface<Other>, imovable>
  constexpr any &operator=(Interface<Other> other) noexcept(_as_large_as<Other>)
  {
    // Guard against self-assignment when other is a reference to *this
    if constexpr (Other::_root_kind == _root_kind::_reference)
      if (data(other) == data(*this))
        return *this;

    reset(*this);
    (*this)._assign(std::move(other));
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
      return other._slice_to(*this);
    }
    else if (other._in_situ())
    {
      return other._slice_to(*this);
    }
    else if ANY_CONSTEVAL
    {
      (*this).pointer = std::exchange(other.pointer, nullptr);
    }
    else
    {
      auto &ptr = *::any::start_lifetime_as<_tagged_ptr>((*this).buffer);
      ptr       = *::any::start_lifetime_as<_tagged_ptr>(other.buffer);
    }
  }

  static_assert(sizeof(iabstract<Interface>) == sizeof(void *)); // sanity check
};

// //////////////////////////////////////////////////////////////////////////////////////////
// // _reference_root
// template <template <class> class Interface, class CvValue>
// struct _indirect_reference_root : iabstract<Interface>
// {
//   static_assert(!extension_of<CvValue, Interface>,
//                 "Value must be a concrete type, not an Interface type.");
//   using interface_type                        = iabstract<Interface>;
//   static constexpr ::any::_box_kind _box_kind = ::any::_box_kind::_object;
//   static constexpr bool _is_reference         = true;

//   constexpr explicit _indirect_reference_root(CvValue &value) noexcept
//     : value_(std::addressof(value))
//   {
//   }

//   [[nodiscard]]
//   constexpr auto &_value() noexcept
//   {
//     return ::any::_unconst(*value_);
//   }

//   [[nodiscard]]
//   constexpr auto &_value() const noexcept
//   {
//     return *value_;
//   }

//   [[nodiscard]]
//   constexpr bool _empty() const noexcept final override
//   {
//     return false;
//   }

//   constexpr void _reset() noexcept final override
//   {
//     if ANY_CONSTEVAL
//     {
//       delete this;
//     }
//   }

//   [[nodiscard]]
//   constexpr type_info const &_type() const noexcept final override
//   {
//     return TYPEID(CvValue);
//   }

//   [[nodiscard]]
//   constexpr void *_data() const noexcept final override
//   {
//     return const_cast<void *>(static_cast<void const *>(value_));
//   }

// private:
//   union
//   {
//     _iroot *pointer; // if active, points to a _value_model
//     CvValue *value_; // if active, points to the referant
//   };
// };

//////////////////////////////////////////////////////////////////////////////////////////
// _reference
template <template <class> class Interface>
struct _reference : Interface<_mcall<_bases_of<Interface>, _reference_proxy_root<Interface>>>
{
};

//////////////////////////////////////////////////////////////////////////////////////////
// _any_ptr_base
template <template <class> class Interface>
struct _any_ptr_base
{
  _any_ptr_base() = default;

  constexpr _any_ptr_base(std::nullptr_t) noexcept
    : ref_()
  {
  }

  constexpr _any_ptr_base(_any_ptr_base const &other) noexcept
    : ref_()
  {
    (*this)._proxy_assign(std::addressof(other.ref_));
  }

  template <template <class> class OtherInterface>
    requires extension_of<iabstract<OtherInterface>, Interface>
  constexpr _any_ptr_base(_any_ptr_base<OtherInterface> const &other) noexcept
    : ref_()
  {
    (*this)._proxy_assign(std::addressof(other.ref_));
  }

  constexpr _any_ptr_base &operator=(_any_ptr_base const &other) noexcept
  {
    reset(ref_);
    (*this)._proxy_assign(std::addressof(other.ref_));
    return *this;
  }

  constexpr _any_ptr_base &operator=(std::nullptr_t) noexcept
  {
    reset(ref_);
    return *this;
  }

  template <template <class> class OtherInterface>
    requires extension_of<iabstract<OtherInterface>, Interface>
  constexpr _any_ptr_base &operator=(_any_ptr_base<OtherInterface> const &other) noexcept
  {
    reset(ref_);
    (*this)._proxy_assign(std::addressof(other.ref_));
    return *this;
  }

  friend constexpr void swap(_any_ptr_base &lhs, _any_ptr_base &rhs) noexcept
  {
    lhs.ref_.swap(rhs.ref_);
  }

  [[nodiscard]]
  constexpr bool operator==(_any_ptr_base const &other) const noexcept
  {
    return data(ref_) == data(other.ref_);
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
    // Optimize for when CvProxy derives from iabstract<Interface>. Store the address of
    // value(other) directly in out as a tagged ptr instead of introducing an indirection.
    else if constexpr (std::derived_from<CvValueProxy, iabstract<Interface>>)
      ref_._proxy_bind(::any::_as_const_if<is_const>(value(*proxy_ptr)));
    else
      value(*proxy_ptr)._indirect_bind(ref_);
  }

  //! @param other A pointer to a reference proxy model implementing Interface.
  template <extension_of<Interface> CvReferenceProxy>
    requires(CvReferenceProxy::_root_kind == _root_kind::_reference)
  constexpr void _proxy_assign(CvReferenceProxy *proxy_ptr) noexcept
  {
    static_assert(CvReferenceProxy::_box_kind == _box_kind::_proxy);
    using model_type        = ::any::_reference_proxy_model<Interface>;
    constexpr bool is_const = std::is_const_v<CvReferenceProxy>;

    if (proxy_ptr == nullptr || empty(*proxy_ptr))
      return;
    // in the case where CvReferenceProxy is a base class of model_type, we can simply
    // downcast and copy the model directly.
    else if constexpr (std::derived_from<model_type, CvReferenceProxy>)
      ref_ = *::any::_polymorphic_downcast<model_type const *>(proxy_ptr);
    // Otherwise, we are assigning from a derived reference to a base reference, and the
    // other reference is indirect (i.e., it holds a _reference_model in its buffer). We
    // need to copy the referant model.
    else if ((*proxy_ptr)._is_indirect())
      value(*proxy_ptr)._indirect_bind(ref_);
    else
      ref_._proxy_bind(::any::_as_const_if<is_const>(value(*proxy_ptr)));
  }

  template <class CvValue>
  constexpr void _value_assign(CvValue *value_ptr) noexcept
  {
    if (value_ptr != nullptr)
      ref_._direct_bind(*value_ptr);
  }

  // the proxy model is mutable so that a const any_ptr can return non-const
  // references from operator-> and operator*.
  mutable _reference_proxy_model<Interface> ref_;
};

//////////////////////////////////////////////////////////////////////////////////////////
// any_ptr
template <template <class> class Interface>
struct any_ptr final : _any_ptr_base<Interface>
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
    reset((*this).ref_);
    (*this)._value_assign(value_ptr);
    return *this;
  }

  template <extension_of<Interface> Proxy>
  constexpr any_ptr &operator=(Proxy *proxy_ptr) noexcept
  {
    reset((*this).ref_);
    (*this)._proxy_assign(proxy_ptr);
    return *this;
  }

  template <extension_of<Interface> Proxy>
  any_ptr &operator=(Proxy const *proxy_ptr) = delete;

  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline constexpr auto *operator->() const noexcept
  {
    return std::addressof((*this).ref_);
  }

  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline constexpr auto &operator*() const noexcept
  {
    return (*this).ref_;
  }
};

template <template <class> class Interface, class Base>
any_ptr(Interface<Base> *) -> any_ptr<Interface>;

//////////////////////////////////////////////////////////////////////////////////////////
// any_const_ptr
template <template <class> class Interface>
struct any_const_ptr final : _any_ptr_base<Interface>
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
    reset((*this).ref_);
    (*this)._value_assign(value_ptr);
    return *this;
  }

  template <extension_of<Interface> Proxy>
  constexpr any_const_ptr &operator=(Proxy const *proxy_ptr) noexcept
  {
    reset((*this).ref_);
    (*this)._proxy_assign(proxy_ptr);
    return *this;
  }

  friend constexpr void swap(any_const_ptr &a, any_const_ptr &b) noexcept
  {
    a.ref_.swap(b.ref_);
  }

  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline constexpr auto const *operator->() const noexcept
  {
    return std::addressof((*this).ref_);
  }

  [[ANY_ALWAYS_INLINE, nodiscard]]
  inline constexpr auto const &operator*() const noexcept
  {
    return (*this).ref_;
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

    if (type == TYPEID(void))
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
