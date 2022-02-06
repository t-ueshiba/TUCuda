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
  \brief	cuda::std::tupleの用途拡張のためのユティリティ
*/
#pragma once

#include <cuda/std/tuple>
#include "TU/tuple.h"
#include <thrust/device_ptr.h>

namespace thrust
{
/************************************************************************
*  thrust::stride(const ITER&)						*
************************************************************************/
template <class T> __host__ __device__ ptrdiff_t
stride(device_ptr<T>)							;

}	// namespace thrust

namespace TU
{
namespace cuda
{
/************************************************************************
*  predicate: is_tuple<T>						*
************************************************************************/
//! 与えられた型が cuda::std::tuple 又はそれに変換可能であるか判定する
/*!
  \param T	判定対象となる型
*/ 
template <class T>
using is_tuple = is_convertible<T, ::cuda::std::tuple>;

/************************************************************************
*  tuple_for_each(FUNC, TUPLES&&...)				`	*
************************************************************************/
namespace detail
{
  template <size_t I, class T, std::enable_if_t<!is_tuple<T>::value>* = nullptr>
  __host__ __device__ inline decltype(auto)
  tuple_get(T&& x)
  {
      return std::forward<T>(x);
  }
  template <size_t I, class T, std::enable_if_t<is_tuple<T>::value>* = nullptr>
  __host__ __device__ inline decltype(auto)
  tuple_get(T&& x)
  {
      return ::cuda::std::get<I>(std::forward<T>(x));
  }

  template <class... T>
  struct first_tuple_size;
  template<>
  struct first_tuple_size<>
  {
      constexpr static size_t	value = 0;
  };
  template <class HEAD, class... TAIL>
  struct first_tuple_size<HEAD, TAIL...>
  {
      template <class T_>
      struct tuple_size
      {
	  constexpr static size_t	value = 0;
      };
      template <class... T_>
      struct tuple_size<::cuda::std::tuple<T_...> >
      {
	  constexpr static size_t	value = sizeof...(T_);
      };

      using TUPLE = std::decay_t<HEAD>;
      
      constexpr static size_t	value = (tuple_size<TUPLE>::value ?
					 tuple_size<TUPLE>::value :
					 first_tuple_size<TAIL...>::value);
  };
    
  template <class FUNC, class... TUPLES> __host__ __device__ inline void
  tuple_for_each(std::index_sequence<>, FUNC&&, TUPLES&&...)
  {
  }
  template <size_t I, size_t... IDX, class FUNC, class... TUPLES>
  __host__ __device__ inline void
  tuple_for_each(std::index_sequence<I, IDX...>, FUNC&& f, TUPLES&&... x)
  {
      f(detail::tuple_get<I>(std::forward<TUPLES>(x))...);
      tuple_for_each(std::index_sequence<IDX...>(),
		     std::forward<FUNC>(f), std::forward<TUPLES>(x)...);
  }
}	// namespace detail
    
template <class FUNC, class... TUPLES>
__host__ __device__ inline std::enable_if_t<any<is_tuple, TUPLES...>::value>
tuple_for_each(FUNC&& f, TUPLES&&... x)
{
    detail::tuple_for_each(std::make_index_sequence<
			       detail::first_tuple_size<TUPLES...>::value>(),
			   std::forward<FUNC>(f), std::forward<TUPLES>(x)...);
}

/************************************************************************
*  tuple_transform(FUNC, TUPLES&&...)					*
************************************************************************/
namespace detail
{
  template <class FUNC, class... TUPLES> __host__ __device__ inline auto
  tuple_transform(std::index_sequence<>, FUNC&&, TUPLES&&...)
  {
      return ::cuda::std::tuple<>();
  }
  template <class FUNC, class... TUPLES, size_t I, size_t... IDX>
  __host__ __device__ inline auto
  tuple_transform(std::index_sequence<I, IDX...>, FUNC&& f, TUPLES&&... x)
  {
      auto&&	val = f(detail::tuple_get<I>(std::forward<TUPLES>(x))...);
      return ::cuda::std::tuple_cat(
		::cuda::std::make_tuple(
		    make_reference_wrapper(std::forward<decltype(val)>(val))),
		detail::tuple_transform(std::index_sequence<IDX...>(),
					std::forward<FUNC>(f),
					std::forward<TUPLES>(x)...));
  }
}	// namespace detail
    
template <class FUNC, class... TUPLES,
	  std::enable_if_t<any<is_tuple, TUPLES...>::value>* = nullptr>
__host__ __device__ inline auto
tuple_transform(FUNC&& f, TUPLES&&... x)
{
    return detail::tuple_transform(
	       std::make_index_sequence<
		   detail::first_tuple_size<TUPLES...>::value>(),
	       std::forward<FUNC>(f), std::forward<TUPLES>(x)...);
}

/************************************************************************
*  Arithmetic operators							*
************************************************************************/
template <class E, std::enable_if_t<is_tuple<E>::value>* = nullptr>
__host__ __device__ inline auto
operator -(E&& expr)
{
    return tuple_transform([](auto&& x)
			   { return -std::forward<decltype(x)>(x); },
			   std::forward<E>(expr));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator +(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  + std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator -(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  - std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator *(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  * std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator /(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  / std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__  inline auto
operator %(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  % std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R>
__host__ __device__ inline std::enable_if_t<is_tuple<L>::value, L&>
operator +=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x += y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ inline std::enable_if_t<is_tuple<L>::value, L&>
operator -=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x -= y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ inline std::enable_if_t<is_tuple<L>::value, L&>
operator *=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x *= y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ inline std::enable_if_t<is_tuple<L>::value, L&>
operator /=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x /= y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ inline std::enable_if_t<is_tuple<L>::value, L&>
operator %=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x %= y; }, l, r);
    return l;
}

template <class T>
__host__ __device__ inline std::enable_if_t<is_tuple<T>::value, T&>
operator ++(T&& t)
{
    tuple_for_each([](auto&& x){ ++x; }, t);
    return t;
}

template <class T>
__host__ __device__ inline std::enable_if_t<is_tuple<T>::value, T&>
operator --(T&& t)
{
    tuple_for_each([](auto&& x){ --x; }, t);
    return t;
}

template <class L, class C, class R,
	  std::enable_if_t<any<is_tuple, L, C, R>::value>* = nullptr>
__host__ __device__ inline auto
fma(L&& l, C&& c, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y, auto&& z)
			   { return fma(std::forward<decltype(x)>(x),
					std::forward<decltype(y)>(y),
					std::forward<decltype(z)>(z)); },
			   std::forward<L>(l),
			   std::forward<C>(c),
			   std::forward<R>(r));
}

/************************************************************************
*  Bit operators							*
************************************************************************/
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator &(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  & std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator |(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  | std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator ^(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  ^ std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}
    
template <class L, class R>
__host__ __device__ inline std::enable_if_t<is_tuple<L>::value, L&>
operator &=(L&& l, const R& r)
{
    tuple_for_each([](auto& x, const auto& y){ x &= y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ inline std::enable_if_t<is_tuple<L>::value, L&>
operator |=(L&& l, const R& r)
{
    tuple_for_each([](auto& x, const auto& y){ x |= y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ inline std::enable_if_t<is_tuple<L>::value, L&>
operator ^=(L&& l, const R& r)
{
    tuple_for_each([](auto& x, const auto& y){ x ^= y; }, l, r);
    return l;
}

/************************************************************************
*  Logical operators							*
************************************************************************/
template <class... T> __host__ __device__ inline auto
operator !(const std::tuple<T...>& t)
{
    return tuple_transform([](const auto& x){ return !x; }, t);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
operator &&(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x && y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr> inline auto
operator ||(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x || y; }, l, r);
}
    
/************************************************************************
*  Relational operators							*
************************************************************************/
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator ==(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x == y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator !=(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x != y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator <(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x < y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator >(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x > y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator <=(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x <= y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator >=(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x >= y; }, l, r);
}

/************************************************************************
*  Selection								*
************************************************************************/
template <class X, class Y> __host__ __device__ inline auto
select(bool s, X&& x, Y&& y)
{
    return (s ? std::forward<X>(x) : std::forward<Y>(y));
}
    
template <class... S, class X, class Y,
	  std::enable_if_t<any<is_tuple, X, Y>::value>* = nullptr>
__host__ __device__ inline auto
select(const std::tuple<S...>& s, X&& x, Y&& y)
{
    return tuple_transform([](const auto& t, auto&& u, auto&& v)
			   { return select(t,
					   std::forward<decltype(u)>(u),
					   std::forward<decltype(v)>(v)); },
			   s, std::forward<X>(x), std::forward<Y>(y));
}

/************************************************************************
*  class zip_iterator<ITER_TUPLE>					*
************************************************************************/
namespace detail
{
  struct generic_dereference
  {
    // assignment_iterator<FUNC, ITER> のように dereference すると
    // その base iterator への参照を内包する proxy を返す反復子もあるので，
    // 引数は const ITER_& 型にする．もしも ITER_ 型にすると，呼出側から
    // コピーされたローカルな反復子 iter への参照を内包する proxy を
    // 返してしまい，dangling reference が生じる．
      template <class ITER_> __host__ __device__
      decltype(auto)	operator ()(const ITER_& iter) const
			{
			    return *iter;
			}
  };
}	// namespace detail

template <class ITER_TUPLE>
class zip_iterator
    : public thrust::iterator_facade<
		zip_iterator<ITER_TUPLE>,
		decltype(cuda::tuple_transform(
			     TU::cuda::detail::generic_dereference(),
			     std::declval<ITER_TUPLE>())),
		thrust::use_default,
		typename std::iterator_traits<
		   ::cuda::std::tuple_element_t<0, ITER_TUPLE> >
		   ::iterator_category,
		decltype(cuda::tuple_transform(
			     TU::cuda::detail::generic_dereference(),
			     std::declval<ITER_TUPLE>()))>
{
  private:
    using super = thrust::iterator_facade<
		        zip_iterator,
			decltype(cuda::tuple_transform(
				     TU::cuda::detail::generic_dereference(),
				     std::declval<ITER_TUPLE>())),
			thrust::use_default,
			typename std::iterator_traits<
			    ::cuda::std::tuple_element_t<0, ITER_TUPLE> >
			    ::iterator_category,
			decltype(cuda::tuple_transform(
				     TU::cuda::detail::generic_dereference(),
				     std::declval<ITER_TUPLE>()))>;
    friend	class thrust::iterator_core_access;
    
  public:
    using	typename super::reference;
    using	typename super::difference_type;
    
  public:
    __host__ __device__
		zip_iterator(ITER_TUPLE iter_tuple)
		    :_iter_tuple(iter_tuple)				{}
    template <class ITER_TUPLE_,
	      std::enable_if_t<
		  std::is_convertible<ITER_TUPLE_, ITER_TUPLE>::value>*
	      = nullptr>
    __host__ __device__
		zip_iterator(const zip_iterator<ITER_TUPLE_>& iter)
		    :_iter_tuple(iter.get_iterator_tuple())		{}

    __host__ __device__
    const auto&	get_iterator_tuple() const
		{
		    return _iter_tuple;
		}
    
  private:
    __host__ __device__
    reference	dereference() const
		{
		    return tuple_transform(detail::generic_dereference(),
					   _iter_tuple);
		}
    template <class ITER_TUPLE_>
    __host__ __device__
    std::enable_if_t<std::is_convertible<ITER_TUPLE_, ITER_TUPLE>::value, bool>
		equal(const zip_iterator<ITER_TUPLE_>& iter) const
		{
		    return ::cuda::std::get<0>(iter.get_iterator_tuple())
			== ::cuda::std::get<0>(_iter_tuple);
		}
    __host__ __device__
    void	increment()
		{
		    ++_iter_tuple;
		}
    __host__ __device__
    void	decrement()
		{
		    --_iter_tuple;
		}
    __host__ __device__
    void	advance(difference_type n)
		{
		    _iter_tuple += n;
		}
    template <class ITER_TUPLE_>
    __host__ __device__
    std::enable_if_t<std::is_convertible<ITER_TUPLE_, ITER_TUPLE>::value,
		     difference_type>
		distance_to(const zip_iterator<ITER_TUPLE_>& iter) const
		{
		    return ::cuda::std::get<0>(iter.get_iterator_tuple())
			 - ::cuda::std::get<0>(_iter_tuple);
		}

  private:
    ITER_TUPLE	_iter_tuple;
};

template <class... ITERS>
__host__ __device__ inline zip_iterator<::cuda::std::tuple<ITERS...> >
make_zip_iterator(const ::cuda::std::tuple<ITERS...>& iter_tuple)
{
    return {iter_tuple};
}

template <class ITER> __host__ __device__ inline ITER
make_zip_iterator(const ITER& iter)
{
    return iter;
}

template <class ITER, class... ITERS> __host__ __device__ inline auto
make_zip_iterator(const ITER& iter, const ITERS&... iters)
{
    return make_zip_iterator(::cuda::std::make_tuple(iter, iters...));
}

/************************************************************************
*  TU::[begin|end|rbegin|rend](TUPLE&&)					*
************************************************************************/
namespace detail
{
  template <class T>
  auto	check_begin(const T& x) -> decltype(begin(x), std::true_type())	;
  auto	check_begin(...)	-> std::false_type			;

  template <class T>
  using has_begin = decltype(check_begin(std::declval<T>()));
}	// namespace detail

template <class... T,
	  std::enable_if_t<all<detail::has_begin, T...>::value>* = nullptr>
inline auto
begin(::cuda::std::tuple<T...>& t)
{
    return make_zip_iterator(
		tuple_transform([](auto& x)
				{ using std::begin; return begin(x); }, t));
}

template <class... T,
	  std::enable_if_t<all<detail::has_begin, T...>::value>* = nullptr>
inline auto
end(::cuda::std::tuple<T...>& t)
{
    return make_zip_iterator(
		tuple_transform([](auto& x)
				{ using std::end; return end(x); }, t));
}

template <class... T> inline auto
rbegin(::cuda::std::tuple<T...>& t)
    -> decltype(std::make_reverse_iterator(end(t)))
{
    return std::make_reverse_iterator(end(t));
}

template <class... T> inline auto
rend(::cuda::std::tuple<T...>& t)
    -> decltype(std::make_reverse_iterator(begin(t)))
{
    return std::make_reverse_iterator(begin(t));
}

template <class... T,
	  std::enable_if_t<all<detail::has_begin, T...>::value>* = nullptr>
inline auto
begin(::cuda::std::tuple<T...>&& t)
{
    return make_zip_iterator(
		tuple_transform([](auto& x)
				{ using std::begin; return begin(x); }, t));
}

template <class... T,
	  std::enable_if_t<all<detail::has_begin, T...>::value>* = nullptr>
inline auto
end(::cuda::std::tuple<T...>&& t)
{
    return make_zip_iterator(
		tuple_transform([](auto& x)
				{ using std::end; return end(x); }, t));
}

template <class... T> inline auto
rbegin(::cuda::std::tuple<T...>&& t)
    -> decltype(std::make_reverse_iterator(end(t)))
{
    return std::make_reverse_iterator(end(t));
}

template <class... T> inline auto
rend(::cuda::std::tuple<T...>&& t)
    -> decltype(std::make_reverse_iterator(begin(t)))
{
    return std::make_reverse_iterator(begin(t));
}

template <class... T,
	  std::enable_if_t<all<detail::has_begin, T...>::value>* = nullptr>
inline auto
begin(const ::cuda::std::tuple<T...>& t)
{
    return make_zip_iterator(
		tuple_transform([](auto& x)
				{ using std::begin; return begin(x); }, t));
}

template <class... T,
	  std::enable_if_t<all<detail::has_begin, T...>::value>* = nullptr>
inline auto
end(const ::cuda::std::tuple<T...>& t)
{
    return make_zip_iterator(
		tuple_transform([](auto& x)
				{ using std::end; return end(x); }, t));
}

template <class... T> inline auto
rbegin(const ::cuda::std::tuple<T...>& t)
    -> decltype(std::make_reverse_iterator(end(t)))
{
    return std::make_reverse_iterator(end(t));
}

template <class... T> inline auto
rend(const ::cuda::std::tuple<T...>& t)
    -> decltype(std::make_reverse_iterator(begin(t)))
{
    return std::make_reverse_iterator(begin(t));
}

template <class... T> inline auto
cbegin(const ::cuda::std::tuple<T...>& t) -> decltype(begin(t))
{
    return begin(t);
}

template <class... T> inline auto
cend(const ::cuda::std::tuple<T...>& t) -> decltype(end(t))
{
    return end(t);
}

template <class... T> inline auto
crbegin(const ::cuda::std::tuple<T...>& t) -> decltype(rbegin(t))
{
    return rbegin(t);
}

template <class... T> inline auto
crend(const ::cuda::std::tuple<T...>& t) -> decltype(rend(t))
{
    return rend(t);
}

/************************************************************************
*  type alias: decayed_iterator_value<ITER>				*
************************************************************************/
namespace detail
{
  template <class ITER>
  struct decayed_iterator_value
  {
      using type = typename std::iterator_traits<ITER>::value_type;
  };
  template <class... ITER>
  struct decayed_iterator_value<TU::cuda::zip_iterator<::cuda::std::tuple<ITER...> > >
  {
      using type = ::cuda::std::tuple<typename detail::decayed_iterator_value<ITER>::type...>;
  };
}	// namespace detail

//! 反復子が指す型を返す．
/*!
  zip_iterator<ITER_TUPLE>::value_type はITER_TUPLE中の各反復子が指す値への
  参照のtupleの型であるが，decayed_iterator_value<zip_iterator<ITER_TUPLE> >
  は，ITER_TUPLE中の各反復子が指す値そのもののtupleの型を返す．
  \param ITER	反復子
*/
template <class ITER>
using decayed_iterator_value = typename TU::cuda::detail::decayed_iterator_value<ITER>
					      ::type;

/************************************************************************
*  Applying a multi-input function to a tuple of arguments		*
************************************************************************/
namespace detail
{
  template <class FUNC, class TUPLE, size_t... IDX> inline decltype(auto)
  apply(FUNC&& f, TUPLE&& t, std::index_sequence<IDX...>)
  {
      return f(::cuda::std::get<IDX>(std::forward<TUPLE>(t))...);
  }
}

//! 複数の引数をまとめたtupleを関数に適用する
/*!
  t が std::tuple でない場合は f を1引数関数とみなして t をそのまま渡す．
  \param f	関数
  \param t	引数をまとめたtuple
  \return	関数の戻り値
*/
template <class FUNC, class TUPLE,
	  std::enable_if_t<is_tuple<TUPLE>::value>* = nullptr>
inline decltype(auto)
apply(FUNC&& f, TUPLE&& t)
{
    return detail::apply(std::forward<FUNC>(f), std::forward<TUPLE>(t),
			 std::make_index_sequence<
			 ::cuda::std::tuple_size<std::decay_t<TUPLE> >::value>());
}
template <class FUNC, class T,
	  std::enable_if_t<!is_tuple<T>::value>* = nullptr>
inline decltype(auto)
apply(FUNC&& f, T&& t)
{
    return f(std::forward<T>(t));
}

/************************************************************************
*  TU::cuda::stride(const ITER&)					*
************************************************************************/
template <class... ITER> __host__ __device__ inline auto
stride(const ::cuda::std::tuple<ITER...>& iter_tuple)
{
    return tuple_transform([](const auto& iter){ return stride(iter); },
			   iter_tuple);
}
	    
template <class ITER_TUPLE> __host__ __device__ inline auto
stride(const zip_iterator<ITER_TUPLE>& iter)
    -> decltype(stride(iter.get_iterator_tuple()))
{
    return stride(iter.get_iterator_tuple());
}

}	// namespace cuda
}	// namespace TU
