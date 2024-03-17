/*!
  \file		tuple.h
  \author	Toshio UESHIBA
  \brief	cuda::std::tupleの用途拡張のためのユティリティ
*/
#pragma once

#include "TU/type_traits.h"	// for TU::any<PRED, T...> and TU::iterable<T>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <thrust/iterator/zip_iterator.h>

namespace TU::cu
{
/************************************************************************
*  predicate: is_tuple<T>						*
************************************************************************/
//! 与えられた型が cuda::std::tuple 又はそれに変換可能であるか判定する
/*!
  \param T	判定対象となる型
*/
template <class T>
using is_tuple = is_convertible<T, cuda::std::tuple>;

/************************************************************************
*  tuple_for_each(FUNC, TUPLES&&...)				`	*
************************************************************************/
namespace detail
{
  template <size_t I, class T, std::enable_if_t<!is_tuple<T>::value>* = nullptr>
  __host__ __device__ __forceinline__ decltype(auto)
  tuple_get(T&& x)
  {
      return cuda::std::forward<T>(x);
  }
  template <size_t I, class T, std::enable_if_t<is_tuple<T>::value>* = nullptr>
  __host__ __device__ __forceinline__ decltype(auto)
  tuple_get(T&& x)
  {
      return cuda::std::get<I>(cuda::std::forward<T>(x));
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
      struct tuple_size<cuda::std::tuple<T_...> >
      {
	  constexpr static size_t	value = sizeof...(T_);
      };

      using TUPLE = std::decay_t<HEAD>;
      
      constexpr static size_t	value = (tuple_size<TUPLE>::value ?
					 tuple_size<TUPLE>::value :
					 first_tuple_size<TAIL...>::value);
  };
    
  template <class FUNC, class... TUPLES>
  __host__ __device__ __forceinline__ void
  tuple_for_each(cuda::std::index_sequence<>, FUNC&&, TUPLES&&...)
  {
  }
  template <size_t I, size_t... IDX, class FUNC, class... TUPLES>
  __host__ __device__ __forceinline__ void
  tuple_for_each(cuda::std::index_sequence<I, IDX...>, FUNC&& f, TUPLES&&... x)
  {
      f(detail::tuple_get<I>(cuda::std::forward<TUPLES>(x))...);
      detail::tuple_for_each(cuda::std::index_sequence<IDX...>(),
			     cuda::std::forward<FUNC>(f),
			     cuda::std::forward<TUPLES>(x)...);
  }
}	// namespace detail
    
template <class FUNC, class... TUPLES>
__host__ __device__ __forceinline__
std::enable_if_t<any<is_tuple, TUPLES...>::value>
tuple_for_each(FUNC&& f, TUPLES&&... x)
{
    detail::tuple_for_each(cuda::std::make_index_sequence<
			       detail::first_tuple_size<TUPLES...>::value>(),
			   cuda::std::forward<FUNC>(f),
			   cuda::std::forward<TUPLES>(x)...);
}

/************************************************************************
*  tuple_transform(FUNC, TUPLES&&...)					*
************************************************************************/
namespace detail
{
  //! 与えられた値の型に応じて実引数を生成する
  /*!
    \param x	関数に渡す引数
    \return	xが右辺値参照ならばx自身，定数参照ならばstd::cref(x),
		非定数参照ならばstd::ref(x)
  */
  template <class T> __host__ __device__ __forceinline__
  std::conditional_t<std::is_lvalue_reference<T>::value,
		     std::reference_wrapper<std::remove_reference_t<T> >, T&&>
  make_reference_wrapper(T&& x)
  {
      return cuda::std::forward<T>(x);
  }
    
  template <class FUNC, class... TUPLES>
  __host__ __device__ __forceinline__ auto
  tuple_transform(std::index_sequence<>, FUNC&&, TUPLES&&...)
  {
      return cuda::std::tuple<>();
  }
  template <class FUNC, class... TUPLES, size_t I, size_t... IDX>
  __host__ __device__ __forceinline__ auto
  tuple_transform(std::index_sequence<I, IDX...>, FUNC&& f, TUPLES&&... x)
  {
      auto&&	val = f(detail::tuple_get<I>(
			    cuda::std::forward<TUPLES>(x))...);
      return cuda::std::tuple_cat(
		 cuda::std::make_tuple(
		    make_reference_wrapper(
			cuda::std::forward<decltype(val)>(val))),
		 detail::tuple_transform(std::index_sequence<IDX...>(),
					 cuda::std::forward<FUNC>(f),
					 cuda::std::forward<TUPLES>(x)...));
  }
}	// namespace detail
    
template <class FUNC, class... TUPLES,
	  std::enable_if_t<any<is_tuple, TUPLES...>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
tuple_transform(FUNC&& f, TUPLES&&... x)
{
    return detail::tuple_transform(
	       std::make_index_sequence<
		   detail::first_tuple_size<TUPLES...>::value>(),
	       cuda::std::forward<FUNC>(f), cuda::std::forward<TUPLES>(x)...);
}

/************************************************************************
*  Arithmetic operators							*
************************************************************************/
template <class E, std::enable_if_t<is_tuple<E>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator -(E&& expr)
{
    return tuple_transform([](auto&& x)
			   { return -cuda::std::forward<decltype(x)>(x); },
			   cuda::std::forward<E>(expr));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator +(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return cuda::std::forward<decltype(x)>(x)
				  + cuda::std::forward<decltype(y)>(y); },
			   cuda::std::forward<L>(l), cuda::std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator -(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return cuda::std::forward<decltype(x)>(x)
				  - cuda::std::forward<decltype(y)>(y); },
			   cuda::std::forward<L>(l), cuda::std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator *(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return cuda::std::forward<decltype(x)>(x)
				  * cuda::std::forward<decltype(y)>(y); },
			   cuda::std::forward<L>(l), cuda::std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator /(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return cuda::std::forward<decltype(x)>(x)
				  / cuda::std::forward<decltype(y)>(y); },
			   cuda::std::forward<L>(l), cuda::std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator %(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return cuda::std::forward<decltype(x)>(x)
				  % cuda::std::forward<decltype(y)>(y); },
			   cuda::std::forward<L>(l), cuda::std::forward<R>(r));
}

template <class L, class R>
__host__ __device__ __forceinline__ std::enable_if_t<is_tuple<L>::value, L&>
operator +=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x += y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ __forceinline__ std::enable_if_t<is_tuple<L>::value, L&>
operator -=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x -= y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ __forceinline__ std::enable_if_t<is_tuple<L>::value, L&>
operator *=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x *= y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ __forceinline__ std::enable_if_t<is_tuple<L>::value, L&>
operator /=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x /= y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ __forceinline__ std::enable_if_t<is_tuple<L>::value, L&>
operator %=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x %= y; }, l, r);
    return l;
}

template <class T>
__host__ __device__ __forceinline__ std::enable_if_t<is_tuple<T>::value, T&>
operator ++(T&& t)
{
    tuple_for_each([](auto&& x){ ++x; }, t);
    return t;
}

template <class T>
__host__ __device__ __forceinline__ std::enable_if_t<is_tuple<T>::value, T&>
operator --(T&& t)
{
    tuple_for_each([](auto&& x){ --x; }, t);
    return t;
}

/************************************************************************
*  Bit operators							*
************************************************************************/
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator &(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  & std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator |(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  | std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator ^(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  ^ std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}
    
template <class L, class R>
__host__ __device__ __forceinline__ std::enable_if_t<is_tuple<L>::value, L&>
operator &=(L&& l, const R& r)
{
    tuple_for_each([](auto& x, const auto& y){ x &= y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ __forceinline__ std::enable_if_t<is_tuple<L>::value, L&>
operator |=(L&& l, const R& r)
{
    tuple_for_each([](auto& x, const auto& y){ x |= y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ __forceinline__ std::enable_if_t<is_tuple<L>::value, L&>
operator ^=(L&& l, const R& r)
{
    tuple_for_each([](auto& x, const auto& y){ x ^= y; }, l, r);
    return l;
}

/************************************************************************
*  Logical operators							*
************************************************************************/
template <class... T> __host__ __device__ __forceinline__ auto
operator !(const std::tuple<T...>& t)
{
    return tuple_transform([](const auto& x){ return !x; }, t);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator &&(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x && y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
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
__host__ __device__ __forceinline__ auto
operator ==(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x == y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator !=(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x != y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator <(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x < y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator >(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x > y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator <=(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x <= y; }, l, r);
}
    
template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator >=(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y)
			   { return x >= y; }, l, r);
}

/************************************************************************
*  Selection								*
************************************************************************/
template <class X, class Y> __host__ __device__ __forceinline__ auto
select(bool s, X&& x, Y&& y)
{
    return (s ? cuda::std::forward<X>(x) : cuda::std::forward<Y>(y));
}
    
template <class... S, class X, class Y,
	  std::enable_if_t<any<is_tuple, X, Y>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
select(const cuda::std::tuple<S...>& s, X&& x, Y&& y)
{
    return tuple_transform([](const auto& t, auto&& u, auto&& v)
			   { return select(
					t,
					cuda::std::forward<decltype(u)>(u),
					cuda::std::forward<decltype(v)>(v)); },
			   s,
			   cuda::std::forward<X>(x), cuda::std::forward<Y>(y));
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
  struct decayed_iterator_value<thrust::zip_iterator<
				    cuda::std::tuple<ITER...> > >
  {
      using type = cuda::std::tuple<
			typename decayed_iterator_value<ITER>::type...>;
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
using decayed_iterator_value = typename detail::decayed_iterator_value<ITER>
					      ::type;

/************************************************************************
*  Applying a multi-input function to a tuple of arguments		*
************************************************************************/
template <class FUNC, class TUPLE,
	  std::enable_if_t<is_tuple<TUPLE>::value>* = nullptr>
__host__ __device__ __forceinline__ decltype(auto)
cu_apply(FUNC&& f, TUPLE&& t)
{
    return cuda::std::apply(cuda::std::forward<FUNC>(f),
			    cuda::std::forward<TUPLE>(t));
}

template <class FUNC, class T,
	  std::enable_if_t<!is_tuple<T>::value>* = nullptr>
__host__ __device__ __forceinline__ decltype(auto)
cu_apply(FUNC&& f, T&& t)
{
    return f(cuda::std::forward<T>(t));
}

}	// namespace TU::cu

/*
 *  cuda::std::tuple<T...> を引数とする以下の関数は，ADLを用いて非修飾名で
 *  呼び出せるように namespace cuda::std で定義する．これにより，
 *  TU::detail::element_t(E&&) や
 *  thrust::stride(const thrust::zip_iterator<ITER_TUPLE>&) 等からの
 *  begin() や stride() の呼び出しに於いて，これらが候補関数となる．
 */
namespace cuda::std
{
/************************************************************************
*  cuda::std::[begin|end|rbegin|rend|size](cuda::std::tuple<T...>&)	*
************************************************************************/
template <class... T,
	  std::enable_if_t<TU::all<TU::is_iterable, T...>::value>*
	  = nullptr>
inline auto
begin(tuple<T...>& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto& x){ using std::begin; return begin(x); }, t));
}

template <class... T,
	  std::enable_if_t<TU::all<TU::is_iterable, T...>::value>*
	  = nullptr>
inline auto
end(tuple<T...>& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto& x){ using std::end; return end(x); }, t));
}

template <class... T> inline auto
rbegin(tuple<T...>& t) -> decltype(::std::make_reverse_iterator(end(t)))
{
    return ::std::make_reverse_iterator(end(t));
}

template <class... T> inline auto
rend(tuple<T...>& t) -> decltype(::std::make_reverse_iterator(begin(t)))
{
    return ::std::make_reverse_iterator(begin(t));
}

template <class... T,
	  std::enable_if_t<TU::all<TU::is_iterable, T...>::value>*
	  = nullptr>
inline auto
begin(tuple<T...>&& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto& x){ using std::begin; return begin(x); }, t));
}

template <class... T,
	  std::enable_if_t<TU::all<TU::is_iterable, T...>::value>*
	  = nullptr>
inline auto
end(tuple<T...>&& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto& x){ using std::end; return end(x); }, t));
}

template <class... T> __host__ __device__ __forceinline__ auto
rbegin(tuple<T...>&& t) -> decltype(::std::make_reverse_iterator(end(t)))
{
    return ::std::make_reverse_iterator(end(t));
}

template <class... T> inline auto
rend(tuple<T...>&& t) -> decltype(::std::make_reverse_iterator(begin(t)))
{
    return ::std::make_reverse_iterator(begin(t));
}

template <class... T,
	  std::enable_if_t<TU::all<TU::is_iterable, T...>::value>*
	  = nullptr>
inline auto
begin(const tuple<T...>& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto& x){ using std::begin; return begin(x); }, t));
}

template <class... T,
	  std::enable_if_t<TU::all<TU::is_iterable, T...>::value>*
	  = nullptr>
inline auto
end(const tuple<T...>& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto& x){ using std::end; return end(x); }, t));
}

template <class... T> inline auto
rbegin(const tuple<T...>& t) -> decltype(::std::make_reverse_iterator(end(t)))
{
    return ::std::make_reverse_iterator(end(t));
}

template <class... T> __host__ __device__ __forceinline__ auto
rend(const tuple<T...>& t) -> decltype(::std::make_reverse_iterator(begin(t)))
{
    return ::std::make_reverse_iterator(begin(t));
}

template <class... T> __host__ __device__ __forceinline__ auto
cbegin(const tuple<T...>& t) -> decltype(begin(t))
{
    return begin(t);
}

template <class... T> inline auto
cend(const tuple<T...>& t) -> decltype(end(t))
{
    return end(t);
}

template <class... T> inline auto
crbegin(const tuple<T...>& t) -> decltype(rbegin(t))
{
    return rbegin(t);
}

template <class... T> inline auto
crend(const tuple<T...>& t) -> decltype(rend(t))
{
    return rend(t);
}

template <class... T> inline std::size_t
size(const tuple<T...>& t)
{
    using	cuda::std::size;
    
    return size(get<0>(t));
}

/************************************************************************
*  cuda::std::stride(const tuple<ITER...>&)				*
************************************************************************/
template <class... ITER> __host__ __device__ __forceinline__ auto
stride(const tuple<ITER...>& iter_tuple)
    -> tuple<decltype(stride(std::declval<ITER>()))...>
{
    return TU::cu::tuple_transform([](const auto& iter)
				   { return stride(iter); },
				   iter_tuple);
}

/************************************************************************
*  I/O functions							*
************************************************************************/
template <class... T> ::std::ostream&
operator <<(::std::ostream& out, const tuple<T...>& t)
{
    out << '(';
    TU::cu::tuple_for_each([&out](const auto& x){ out << ' ' << x; }, t);
    out << ')';

    return out;
}
}	// namespace cuda::std
