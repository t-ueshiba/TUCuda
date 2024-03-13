/*!
  \file		tuple.h
  \author	Toshio UESHIBA
  \brief	cuda::std::tupleの用途拡張のためのユティリティ
*/
#pragma once

#include "TU/tuple.h"	// for TU::any<PRED, T...> and detail::has_begin
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <thrust/device_ptr.h>
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
*  type alias: replace_element<S, T>					*
************************************************************************/
namespace detail
{
  template <class S, class T>
  struct replace_element : std::conditional<std::is_void<T>::value, S, T>
  {
  };
  template <class... S, class T>
  struct replace_element<cuda::std::tuple<S...>, T>
  {
      using type = cuda::std::tuple<typename replace_element<S, T>::type...>;
  };
}	// namespace detail
    
//! 与えられた型がcuda::std::tupleならばその要素の型を，そうでなければ元の型自身を別の型で置き換える．
/*!
  \param S	要素型置換の対象となる型
  \param T	置換後の要素の型．voidならば置換しない．
*/
template <class S, class T>
using replace_element = typename detail::replace_element<S, T>::type;

/************************************************************************
*  make_reference_wrapper(T&&)						*
************************************************************************/
//! 与えられた値の型に応じて実引数を生成する
/*!
  \param x	関数に渡す引数
  \return	xが右辺値参照ならばx自身，定数参照ならばstd::cref(x),
		非定数参照ならばstd::ref(x)
*/
template <class T>
inline std::conditional_t<std::is_lvalue_reference<T>::value,
			  std::reference_wrapper<std::remove_reference_t<T> >,
			  T&&>
make_reference_wrapper(T&& x)
{
    return std::forward<T>(x);
}
    
/************************************************************************
*  tuple_for_each(FUNC, TUPLES&&...)				`	*
************************************************************************/
namespace detail
{
  template <size_t I, class T, std::enable_if_t<!is_tuple<T>::value>* = nullptr>
  inline decltype(auto)
  tuple_get(T&& x)
  {
      return std::forward<T>(x);
  }
  template <size_t I, class T, std::enable_if_t<is_tuple<T>::value>* = nullptr>
  inline decltype(auto)
  tuple_get(T&& x)
  {
      return cuda::std::get<I>(std::forward<T>(x));
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
    
  template <class FUNC, class... TUPLES> inline void
  tuple_for_each(std::index_sequence<>, FUNC&&, TUPLES&&...)
  {
  }
  template <size_t I, size_t... IDX, class FUNC, class... TUPLES> inline void
  tuple_for_each(std::index_sequence<I, IDX...>, FUNC&& f, TUPLES&&... x)
  {
      f(detail::tuple_get<I>(std::forward<TUPLES>(x))...);
      detail::tuple_for_each(std::index_sequence<IDX...>(),
			     std::forward<FUNC>(f), std::forward<TUPLES>(x)...);
  }
}	// namespace detail
    
template <class FUNC, class... TUPLES>
inline std::enable_if_t<any<is_tuple, TUPLES...>::value>
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
  template <class FUNC, class... TUPLES> inline auto
  tuple_transform(std::index_sequence<>, FUNC&&, TUPLES&&...)
  {
      return cuda::std::tuple<>();
  }
  template <class FUNC, class... TUPLES, size_t I, size_t... IDX> inline auto
  tuple_transform(std::index_sequence<I, IDX...>, FUNC&& f, TUPLES&&... x)
  {
      auto&&	val = f(detail::tuple_get<I>(std::forward<TUPLES>(x))...);
      return cuda::std::tuple_cat(
		 cuda::std::make_tuple(
		    make_reference_wrapper(std::forward<decltype(val)>(val))),
		 detail::tuple_transform(std::index_sequence<IDX...>(),
					 std::forward<FUNC>(f),
					 std::forward<TUPLES>(x)...));
  }
}	// namespace detail
    
template <class FUNC, class... TUPLES,
	  std::enable_if_t<any<is_tuple, TUPLES...>::value>* = nullptr>
inline auto
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
template <class E, std::enable_if_t<is_tuple<E>::value>* = nullptr> inline auto
operator -(E&& expr)
{
    return tuple_transform([](auto&& x)
			   { return -std::forward<decltype(x)>(x); },
			   std::forward<E>(expr));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator +(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  + std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator -(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  - std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator *(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  * std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator /(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  / std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
}

template <class L, class R,
	  std::enable_if_t<any<is_tuple, L, R>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
operator %(L&& l, R&& r)
{
    return tuple_transform([](auto&& x, auto&& y)
			   { return std::forward<decltype(x)>(x)
				  % std::forward<decltype(y)>(y); },
			   std::forward<L>(l), std::forward<R>(r));
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
  inline std::enable_if_t<!cu::is_tuple<T>::value, std::ostream&>
  print(std::ostream& out, const T& x)
  {
      return out << x;
  }
  template <class... T> inline std::ostream&
  print(std::ostream& out, const cuda::std::tuple<T...>& x)
  {
      return print(print(out << ' ', cuda::std::get<0>(x)), get_tail(x));
  }

  template <class... T> inline std::ostream&
  operator <<(std::ostream& out, const cuda::std::tuple<T...>& x)
  {
      return print(out << '(', x) << ')';
  }
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
  struct decayed_iterator_value<thrust::zip_iterator<cuda::std::tuple<ITER...> > >
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
    return cuda::std::apply(std::forward<FUNC>(f), std::forward<TUPLE>(t));
}

template <class FUNC, class T,
	  std::enable_if_t<!is_tuple<T>::value>* = nullptr>
__host__ __device__ __forceinline__ decltype(auto)
cu_apply(FUNC&& f, T&& t)
{
    return f(std::forward<T>(t));
}

}	// namespace TU::cu

namespace cuda::std
{
/************************************************************************
*  cuda::std::[begin|end|rbegin|rend|size](cuda::std::tuple<T...>&)	*
************************************************************************/
template <class... T,
	  std::enable_if_t<TU::all<TU::detail::has_begin, T...>::value>*
	  = nullptr>
inline auto
begin(tuple<T...>& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto& x){ using std::begin; return begin(x); }, t));
}

template <class... T,
	  std::enable_if_t<TU::all<TU::detail::has_begin, T...>::value>*
	  = nullptr>
inline auto
end(tuple<T...>& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto& x){ using std::end; return end(x); }, t));
}

template <class... T> inline auto
rbegin(tuple<T...>& t) -> decltype(std::make_reverse_iterator(end(t)))
{
    return std::make_reverse_iterator(end(t));
}

template <class... T> inline auto
rend(tuple<T...>& t) -> decltype(std::make_reverse_iterator(begin(t)))
{
    return std::make_reverse_iterator(begin(t));
}

template <class... T,
	  std::enable_if_t<TU::all<TU::detail::has_begin, T...>::value>*
	  = nullptr>
inline auto
begin(tuple<T...>&& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto& x){ using std::begin; return begin(x); }, t));
}

template <class... T,
	  std::enable_if_t<TU::all<TU::detail::has_begin, T...>::value>*
	  = nullptr>
inline auto
end(tuple<T...>&& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto& x){ using std::end; return end(x); }, t));
}

template <class... T> __host__ __device__ __forceinline__ auto
rbegin(tuple<T...>&& t) -> decltype(std::make_reverse_iterator(end(t)))
{
    return std::make_reverse_iterator(end(t));
}

template <class... T> inline auto
rend(tuple<T...>&& t) -> decltype(std::make_reverse_iterator(begin(t)))
{
    return std::make_reverse_iterator(begin(t));
}

template <class... T,
	  std::enable_if_t<TU::all<TU::detail::has_begin, T...>::value>*
	  = nullptr>
inline auto
begin(const tuple<T...>& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto& x){ using std::begin; return begin(x); }, t));
}

template <class... T,
	  std::enable_if_t<TU::all<TU::detail::has_begin, T...>::value>*
	  = nullptr>
inline auto
end(const tuple<T...>& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto& x){ using std::end; return end(x); }, t));
}

template <class... T> inline auto
rbegin(const tuple<T...>& t)
    -> decltype(std::make_reverse_iterator(end(t)))
{
    return std::make_reverse_iterator(end(t));
}

template <class... T> __host__ __device__ __forceinline__ auto
rend(const tuple<T...>& t) -> decltype(std::make_reverse_iterator(begin(t)))
{
    return std::make_reverse_iterator(begin(t));
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
    using	std::size;
    
    return size(get<0>(t));
}

/************************************************************************
*  cuda::std::stride(const ITER&)					*
************************************************************************/
template <class... T> __host__ __device__ __forceinline__ auto
stride(const tuple<T...>& iter_tuple)
{
    return TU::cu::tuple_transform([](const auto& iter)
				   { return stride(iter); },
				   iter_tuple);
}

}	// namespace cuda::std

namespace thrust
{
/************************************************************************
*  thrust::stride(const ITER&)						*
************************************************************************/
template <class T> __host__ __device__ ptrdiff_t
stride(device_ptr<T>)							;

template <class ITER_TUPLE> __host__ __device__ __forceinline__ auto
stride(const zip_iterator<ITER_TUPLE>& iter)
    -> decltype(stride(iter.get_iterator_tuple()))
{
    return stride(iter.get_iterator_tuple());
}

}	// namespace thrust
