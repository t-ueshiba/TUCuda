/*!
  \file		tuple.h
  \author	Toshio UESHIBA
  \brief	cuda::std::tupleの用途拡張のためのユティリティ
*/
#pragma once

#include "TU/type_traits.h"	// for TU::any<PRED, T...>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>

namespace TU
{
namespace cu
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
*  type alias: tuple_head<T>						*
************************************************************************/
namespace detail
{
  template <class HEAD, class... TAIL>
  HEAD	tuple_head(const cuda::std::tuple<HEAD, TAIL...>&)		;
  template <class T>
  T	tuple_head(const T&)						;
}	// namespace detail
    
//! 与えられた型がtupleならばその先頭要素の型を，そうでなければ元の型を返す．
/*! 
  \param T	その先頭要素の型を調べるべき型
*/
template <class T>
using tuple_head = decltype(detail::tuple_head(std::declval<T>()));

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
  __host__ __device__ __forceinline__ decltype(auto)
  tuple_get(T&& x)
  {
      return std::forward<T>(x);
  }
  template <size_t I, class T, std::enable_if_t<is_tuple<T>::value>* = nullptr>
  __host__ __device__ __forceinline__ decltype(auto)
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
    
  template <class FUNC, class... TUPLES>
  __host__ __device__ __forceinline__ void
  tuple_for_each(std::index_sequence<>, FUNC&&, TUPLES&&...)
  {
  }
  template <size_t I, size_t... IDX, class FUNC, class... TUPLES>
  __host__ __device__ __forceinline__ void
  tuple_for_each(std::index_sequence<I, IDX...>, FUNC&& f, TUPLES&&... x)
  {
      f(tuple_get<I>(std::forward<TUPLES>(x))...);
      tuple_for_each(std::index_sequence<IDX...>(),
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
      auto&&	val = f(tuple_get<I>(std::forward<TUPLES>(x))...);
      return cuda::std::tuple_cat(
		 cuda::std::make_tuple(
		    make_reference_wrapper(std::forward<decltype(val)>(val))),
		tuple_transform(std::index_sequence<IDX...>(),
				std::forward<FUNC>(f),
				std::forward<TUPLES>(x)...));
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
  struct decayed_iterator_value<zip_iterator<cuda::std::tuple<ITER...> > >
  {
      using type = cuda::std::tuple<typename decayed_iterator_value<ITER>::type...>;
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
apply(FUNC&& f, TUPLE&& t)
{
    return cuda::std::apply(std::forward<FUNC>(f), std::forward<TUPLE>(t));
}

template <class FUNC, class T,
	  std::enable_if_t<!is_tuple<T>::value>* = nullptr>
__host__ __device__ __forceinline__ decltype(auto)
apply(FUNC&& f, T&& t)
{
    return f(std::forward<T>(t));
}

}	// namespace cu
}	// namesrace TU

namespace cuda::std
{
/************************************************************************
*  TU::[begin|end|rbegin|rend|size](TUPLE&&)				*
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
	  std::enable_if_t<TU::all<detail::has_begin, T...>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
begin(cuda::std::tuple<T...>& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto& x){ using std::begin; return begin(x); }, t));
}

template <class... T,
	  std::enable_if_t<TU::all<detail::has_begin, T...>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
end(cuda::std::tuple<T...>& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto& x){ using std::end; return end(x); }, t));
}

template <class... T> __host__ __device__ __forceinline__ auto
rbegin(cuda::std::tuple<T...>& t) -> decltype(std::make_reverse_iterator(end(t)))
{
    return std::make_reverse_iterator(end(t));
}

template <class... T> __host__ __device__ __forceinline__ auto
rend(cuda::std::tuple<T...>& t) -> decltype(std::make_reverse_iterator(begin(t)))
{
    return std::make_reverse_iterator(begin(t));
}

template <class... T,
	  std::enable_if_t<TU::all<detail::has_begin, T...>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
begin(cuda::std::tuple<T...>&& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto& x){ using std::begin; return begin(x); }, t));
}

template <class... T,
	  std::enable_if_t<TU::all<detail::has_begin, T...>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
end(cuda::std::tuple<T...>&& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto& x){ using std::end; return end(x); }, t));
}

template <class... T> __host__ __device__ __forceinline__ auto
rbegin(cuda::std::tuple<T...>&& t) -> decltype(std::make_reverse_iterator(end(t)))
{
    return std::make_reverse_iterator(end(t));
}

template <class... T> __host__ __device__ __forceinline__ auto
rend(cuda::std::tuple<T...>&& t) -> decltype(std::make_reverse_iterator(begin(t)))
{
    return std::make_reverse_iterator(begin(t));
}

template <class... T,
	  std::enable_if_t<TU::all<detail::has_begin, T...>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
begin(const cuda::std::tuple<T...>& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto& x){ using std::begin; return begin(x); }, t));
}

template <class... T,
	  std::enable_if_t<TU::all<detail::has_begin, T...>::value>* = nullptr>
__host__ __device__ __forceinline__ auto
end(const cuda::std::tuple<T...>& t)
{
    return thrust::make_zip_iterator(
		TU::cu::tuple_transform(
		    [](auto& x){ using std::end; return end(x); }, t));
}

template <class... T> __host__ __device__ __forceinline__ auto
rbegin(const cuda::std::tuple<T...>& t)
    -> decltype(std::make_reverse_iterator(end(t)))
{
    return std::make_reverse_iterator(end(t));
}

template <class... T> __host__ __device__ __forceinline__ auto
rend(const cuda::std::tuple<T...>& t)
    -> decltype(std::make_reverse_iterator(begin(t)))
{
    return std::make_reverse_iterator(begin(t));
}

template <class... T> __host__ __device__ __forceinline__ auto
cbegin(const cuda::std::tuple<T...>& t) -> decltype(begin(t))
{
    return begin(t);
}

template <class... T> __host__ __device__ __forceinline__ auto
cend(const cuda::std::tuple<T...>& t) -> decltype(end(t))
{
    return end(t);
}

template <class... T> __host__ __device__ __forceinline__ auto
crbegin(const cuda::std::tuple<T...>& t) -> decltype(rbegin(t))
{
    return rbegin(t);
}

template <class... T> __host__ __device__ __forceinline__ auto
crend(const cuda::std::tuple<T...>& t) -> decltype(rend(t))
{
    return rend(t);
}

template <class... T> __host__ __device__ __forceinline__ std::size_t
size(const cuda::std::tuple<T...>& t)
{
    using	std::size;
    
    return size(std::get<0>(t));
}

}	// namespace cuda::std

namespace thrust
{
/************************************************************************
*  thrust::stride(const ITER&)						*
************************************************************************/
template <class T> __host__ __device__ ptrdiff_t
stride(device_ptr<T>)							;

template <class... T> __host__ __device__ __forceinline__ auto
stride(const tuple<T...>& iter_tuple)
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
