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

#include <cassert>
#include <cstdlib>

#define ANY_PP_STRINGIZE(_ARG) #_ARG

#define ANY_PP_LBRACKET2 [[
#define ANY_PP_RBRACKET2 ]]

#if defined(__clang__)
#  define ANY_COMPILER_CLANG __clang_major__ * 100 + __clang_minor__
#elif defined(__GNUC__)
#  define ANY_COMPILER_GCC __GNUC__ * 100 + __GNUC_MINOR__
#elif defined(_MSC_VER)
#  define ANY_COMPILER_MSVC _MSC_VER
#else
#  error "Unsupported compiler"
#endif

// Define the pragma for the host compiler
#if defined(_MSC_VER)
#  define ANY_PRAGMA(_ARG) __pragma(_ARG)
#else
#  define ANY_PRAGMA(_ARG) _Pragma(ANY_PP_STRINGIZE(_ARG))
#endif

#if defined(__NVCOMPILER)
#  define ANY_DIAG_PUSH                     ANY_PRAGMA(diagnostic push)
#  define ANY_DIAG_POP                      ANY_PRAGMA(diagnostic pop)
#  define ANY_DIAG_SUPPRESS_NVHPC(_WARNING) ANY_PRAGMA(diag_suppress _WARNING)
#elif defined(__clang__)
#  define ANY_DIAG_PUSH                     ANY_PRAGMA(clang diagnostic push)
#  define ANY_DIAG_POP                      ANY_PRAGMA(clang diagnostic pop)
#  define ANY_DIAG_SUPPRESS_CLANG(_WARNING) ANY_PRAGMA(clang diagnostic ignored _WARNING)
#elif defined(__GNUC__)
#  define ANY_DIAG_PUSH                   ANY_PRAGMA(GCC diagnostic push)
#  define ANY_DIAG_POP                    ANY_PRAGMA(GCC diagnostic pop)
#  define ANY_DIAG_SUPPRESS_GCC(_WARNING) ANY_PRAGMA(GCC diagnostic ignored _WARNING)
#elif defined(_MSC_VER)
#  define ANY_DIAG_PUSH                    ANY_PRAGMA(warning(push))
#  define ANY_DIAG_POP                     ANY_PRAGMA(warning(pop))
#  define ANY_DIAG_SUPPRESS_MSVC(_WARNING) ANY_PRAGMA(warning(disable : _WARNING))
#else
#  define ANY_DIAG_PUSH
#  define ANY_DIAG_POP
#endif

#if !defined(ANY_DIAG_SUPPRESS_CLANG)
#  define ANY_DIAG_SUPPRESS_CLANG(_WARNING)
#endif

#if !defined(ANY_DIAG_SUPPRESS_GCC)
#  define ANY_DIAG_SUPPRESS_GCC(_WARNING)
#endif

#if !defined(ANY_DIAG_SUPPRESS_NVHPC)
#  define ANY_DIAG_SUPPRESS_NVHPC(_WARNING)
#endif

#if !defined(ANY_DIAG_SUPPRESS_MSVC)
#  define ANY_DIAG_SUPPRESS_MSVC(_WARNING)
#endif

#if __cpp_rtti || _MSC_VER // MSVC has the typeid operator even with RTTI off
#  include <typeinfo>      // IWYU pragma: keep
#  define ANY_HAS_TYPEID 1
#else
#  define ANY_HAS_TYPEID 0
#endif

#if defined(__clang__)
#  define ANY_ALWAYS_INLINE gnu::always_inline, gnu::artificial, gnu::nodebug
#elif defined(__GNUC__)
#  define ANY_ALWAYS_INLINE gnu::always_inline, gnu::artificial
#elif defined(_MSC_VER)
#  define ANY_ALWAYS_INLINE ANY_PP_RBRACKET2 __forceinline ANY_PP_LBRACKET2
#else
#  define ANY_ALWAYS_INLINE
#endif

#if defined(_MSC_VER)
#  define ANY_EMPTY_BASES ANY_PP_RBRACKET2 __declspec(empty_bases) ANY_PP_LBRACKET2
#else
#  define ANY_EMPTY_BASES
#endif

#if __cpp_auto_cast
#  define ANY_DECAY_COPY(...) auto(__VA_ARGS__)
#else
#  define ANY_DECAY_COPY(...) ::any::_decay_copy(__VA_ARGS__)
#endif

#define ANY_ASSERT(...)                                                                            \
  do                                                                                               \
  {                                                                                                \
    if consteval                                                                                   \
    {                                                                                              \
      if (!(__VA_ARGS__))                                                                          \
        ::any::_die(ANY_PP_STRINGIZE(__VA_ARGS__));                                                \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
      assert(__VA_ARGS__);                                                                         \
    }                                                                                              \
  } while (false)
