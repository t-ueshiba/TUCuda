// Software License Agreement (BSD License)
//
// Copyright (c) 2021, National Institute of Advanced Industrial Science and Technology (AIST)
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//  * Neither the name of National Institute of Advanced Industrial
//    Science and Technology (AIST) nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Toshio Ueshiba
//
/*!
  \file		tuple.h
  \author	Toshio UESHIBA
  \brief	thrust::tupleの用途拡張のためのユティリティ
*/
#pragma once

#include "TU/type_traits.h"
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>

namespace TU
{
namespace cu
{
/************************************************************************
*  predicate: is_cons<T>						*
************************************************************************/
namespace detail
{
  template <class HEAD, class TAIL>
  std::true_type	check_cons(thrust::detail::cons<HEAD, TAIL>)	;
  std::false_type	check_cons(...)					;
}	// namespace detail

//! 与えられた型が thrust::tuple 又はそれに変換可能であるか判定する
/*!
  \param T	判定対象となる型
*/
template <class T>
using is_cons = decltype(detail::check_cons(std::declval<T>()));

/************************************************************************
*  predicate: is_null<T>						*
************************************************************************/
template <class T>
using is_null = std::is_convertible<T, thrust::null_type>;

/************************************************************************
*  tuple_for_each(TUPLES..., FUNC)				`	*
************************************************************************/
namespace detail
{
  template <class T, std::enable_if_t<!is_cons<T>::value>* = nullptr>
  __host__ __device__ __forceinline__ decltype(auto)
  get_head(T&& x)
  {
      return std::forward<T>(x);
  }
  template <class T, std::enable_if_t<is_cons<T>::value>* = nullptr>
  __host__ __device__ __forceinline__ decltype(auto)
  get_head(T&& x)
  {
      return x.get_head();
  }

  template <class T, std::enable_if_t<!is_cons<T>::value>* = nullptr>
  __host__ __device__ __forceinline__ decltype(auto)
  get_tail(T&& x)
  {
      return std::forward<T>(x);
  }
  template <class T, std::enable_if_t<is_cons<T>::value>* = nullptr>
  __host__ __device__ __forceinline__ decltype(auto)
  get_tail(T&& x)
  {
      return x.get_tail();
  }
}	// namespace detail

template <class FUNC, class... TUPLES> __host__ __device__ __forceinline__
std::enable_if_t<any<is_null, TUPLES...>::value>
tuple_for_each(FUNC, TUPLES&&...)
{
}

template <class FUNC, class... TUPLES> __host__ __device__ __forceinline__
std::enable_if_t<any<is_cons, TUPLES...>::value>
tuple_for_each(FUNC f, TUPLES&&... x)
{
    f(detail::get_head(std::forward<TUPLES>(x))...);
    tuple_for_each(f, detail::get_tail(std::forward<TUPLES>(x))...);
}

/************************************************************************
*  tuple_transform(TUPLES..., FUNC)					*
************************************************************************/
namespace detail
{
  template <class HEAD, class TAIL> __host__ __device__ __forceinline__ auto
  make_cons(HEAD&& head, TAIL&& tail)
  {
      return thrust::detail::cons<HEAD, TAIL>(std::forward<HEAD>(head),
					      std::forward<TAIL>(tail));
  }
}	// namespace detail

template <class FUNC, class... TUPLES> __host__ __device__ __forceinline__
std::enable_if_t<!any<is_cons, TUPLES...>::value, thrust::null_type>
tuple_transform(FUNC, TUPLES&&...)
{
    return thrust::null_type();
}
template <class FUNC, class... TUPLES,
	  std::enable_if_t<any<is_cons, TUPLES...>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
tuple_transform(FUNC f, TUPLES&&... x)
{
    return detail::make_cons(f(detail::get_head(std::forward<TUPLES>(x))...),
			     tuple_transform(
				 f,
				 detail::get_tail(std::forward<TUPLES>(x))...));
}

/************************************************************************
*  Arithmetic operators							*
************************************************************************/
template <class T, std::enable_if_t<cu::is_cons<T>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator -(const T& t)
{
    return cu::tuple_transform([](const auto& x){ return -x; }, t);
}

template <class L, class R,
	  std::enable_if_t<any<cu::is_cons, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator +(const L& l, const R& r)
{
    return cu::tuple_transform([](const auto& x, const auto& y)
			       { return x + y; }, l, r);
}

template <class L, class R,
	  std::enable_if_t<any<cu::is_cons, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator -(const L& l, const R& r)
{
    return cu::tuple_transform([](const auto& x, const auto& y)
			       { return x - y; }, l, r);
}

template <class L, class R,
	  std::enable_if_t<any<cu::is_cons, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator *(const L& l, const R& r)
{
    return cu::tuple_transform([](const auto& x, const auto& y)
			       { return x * y; }, l, r);
}

template <class L, class R,
	  std::enable_if_t<any<cu::is_cons, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator /(const L& l, const R& r)
{
    return cu::tuple_transform([](const auto& x, const auto& y)
			       { return x / y; }, l, r);
}

template <class L, class R,
	  std::enable_if_t<any<cu::is_cons, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator %(const L& l, const R& r)
{
    return cu::tuple_transform([](const auto& x, const auto& y)
			       { return x % y; }, l, r);
}

template <class L, class R>
__host__ __device__ __forceinline__ std::enable_if_t<cu::is_cons<L>::value, L&>
operator +=(L&& l, const R& r)
{
    cu::tuple_for_each([](auto&& x, const auto& y){ x += y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ __forceinline__ std::enable_if_t<cu::is_cons<L>::value, L&>
operator -=(L&& l, const R& r)
{
    cu::tuple_for_each([](auto&& x, const auto& y){ x -= y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ __forceinline__ std::enable_if_t<cu::is_cons<L>::value, L&>
operator *=(L&& l, const R& r)
{
    cu::tuple_for_each([](auto&& x, const auto& y){ x *= y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ __forceinline__ std::enable_if_t<cu::is_cons<L>::value, L&>
operator /=(L&& l, const R& r)
{
    cu::tuple_for_each([](auto&& x, const auto& y){ x /= y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ __forceinline__ std::enable_if_t<cu::is_cons<L>::value, L&>
operator %=(L&& l, const R& r)
{
    cu::tuple_for_each([](auto&& x, const auto& y){ x %= y; }, l, r);
    return l;
}

template <class T>
__host__ __device__ __forceinline__ std::enable_if_t<cu::is_cons<T>::value, T&>
operator ++(T&& t)
{
    cu::tuple_for_each([](auto&& x){ ++x; }, t);
    return t;
}

template <class T>
__host__ __device__ __forceinline__ std::enable_if_t<cu::is_cons<T>::value, T&>
operator --(T&& t)
{
    cu::tuple_for_each([](auto&& x){ --x; }, t);
    return t;
}

/************************************************************************
*  I/O functions							*
************************************************************************/
namespace detail
{
  inline std::ostream&
  print(std::ostream& out, thrust::null_type)
  {
      return out;
  }
  template <class T>
  inline std::enable_if_t<!cu::is_cons<T>::value, std::ostream&>
  print(std::ostream& out, const T& x)
  {
      return out << x;
  }
  template <class HEAD, class TAIL> inline std::ostream&
  print(std::ostream& out, const thrust::detail::cons<HEAD, TAIL>& x)
  {
      return print(print(out << ' ', get_head(x)), get_tail(x));
  }

  template <class HEAD, class TAIL> inline std::ostream&
  operator <<(std::ostream& out, const thrust::detail::cons<HEAD, TAIL>& x)
  {
      return print(out << '(', x) << ')';
  }
}

/************************************************************************
*  Applying a multi-input function to a tuple of arguments		*
************************************************************************/
namespace detail
{
  template <class FUNC, class TUPLE, size_t... IDX>
  __host__ __device__ __forceinline__ decltype(auto)
  cu_apply(FUNC&& f, TUPLE&& t, std::index_sequence<IDX...>)
  {
      return f(thrust::get<IDX>(std::forward<TUPLE>(t))...);
  }
}

template <class FUNC, class TUPLE,
	  std::enable_if_t<cu::is_cons<TUPLE>::value>* = nullptr>
__host__ __device__ __forceinline__ decltype(auto)
cu_apply(FUNC&& f, TUPLE&& t)
{
    return detail::cu_apply(
		std::forward<FUNC>(f), std::forward<TUPLE>(t),
		std::make_index_sequence<
			thrust::tuple_size<std::decay_t<TUPLE> >::value>());
}
template <class FUNC, class T,
	  std::enable_if_t<!cu::is_cons<T>::value>* = nullptr>
__host__ __device__ __forceinline__ decltype(auto)
cu_apply(FUNC&& f, T&& t)
{
    return f(std::forward<T>(t));
}

}	// namespace cu
}	// namespace TU

namespace thrust
{
/************************************************************************
*  begin|end|rbegin|rend|size for tuples				*
************************************************************************/
template <class HEAD, class TAIL> inline auto
begin(thrust::detail::cons<HEAD, TAIL>& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto&& x){ using std::begin; return begin(x); }, t));
}

template <class HEAD, class TAIL> inline auto
end(thrust::detail::cons<HEAD, TAIL>& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto&& x){ using std::end; return end(x); }, t));
}

template <class HEAD, class TAIL> inline auto
rbegin(thrust::detail::cons<HEAD, TAIL>& t)
{
    return std::make_reverse_iterator(end(t));
}

template <class HEAD, class TAIL> inline auto
rend(thrust::detail::cons<HEAD, TAIL>& t)
{
    return std::make_reverse_iterator(begin(t));
}

template <class HEAD, class TAIL> inline auto
begin(const thrust::detail::cons<HEAD, TAIL>& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto&& x){ using std::begin; return begin(x); }, t));
}

template <class HEAD, class TAIL> inline auto
end(const thrust::detail::cons<HEAD, TAIL>& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto&& x){ using std::end; return end(x); }, t));
}

template <class HEAD, class TAIL> inline auto
rbegin(const thrust::detail::cons<HEAD, TAIL>& t)
{
    return std::make_reverse_iterator(end(t));
}

template <class HEAD, class TAIL> inline auto
rend(const thrust::detail::cons<HEAD, TAIL>& t)
{
    return std::make_reverse_iterator(begin(t));
}

template <class HEAD, class TAIL> inline size_t
size(const thrust::detail::cons<HEAD, TAIL>& t)
{
    return size(thrust::get<0>(t));
}

/************************************************************************
*  thrust::stride(const ITER&)						*
************************************************************************/
template <class T> __host__ __device__ ptrdiff_t
stride(device_ptr<T>)							;

template <class HEAD, class TAIL> __host__ __device__ __forceinline__ auto
stride(const detail::cons<HEAD, TAIL>& iter_tuple)
{
    return TU::cu::tuple_transform([](const auto& iter)
				     { return stride(iter); }, iter_tuple);
}
	    
template <class ITER_TUPLE> __host__ __device__ __forceinline__ auto
stride(const zip_iterator<ITER_TUPLE>& iter)
    -> decltype(stride(iter.get_iterator_tuple()))
{
    return stride(iter.get_iterator_tuple());
}

}	// namespace thrust
