/*!
  \file		tuple.h
  \author	Toshio UESHIBA
  \brief	cuda::std::tupleの用途拡張のためのユティリティ
*/
#pragma once

#include "TU/tuple.h"	// for TU::any<PRED, T...> and detail::has_begin
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
*  make_reference_wrapper(T&&)						*
************************************************************************/
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

#if 0
/************************************************************************
*  class TU::cu::zip_iterator<ITER_TUPLE>				*
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
class zip_iterator : public thrust::iterator_facade<
			zip_iterator<ITER_TUPLE>,
			decltype(tuple_transform(detail::generic_dereference(),
						 std::declval<ITER_TUPLE>())),
			thrust::use_default,
			typename cuda::std::iterator_traits<
			    cuda::std::tuple_element_t<0, ITER_TUPLE> >
			        ::iterator_category,
			decltype(tuple_transform(detail::generic_dereference(),
						 std::declval<ITER_TUPLE>()))>
{
  private:
    using super = thrust::iterator_facade<
			zip_iterator,
			decltype(tuple_transform(detail::generic_dereference(),
						 std::declval<ITER_TUPLE>())),
			thrust::use_default,
			typename cuda::std::iterator_traits<
			    cuda::std::tuple_element_t<0, ITER_TUPLE> >
			        ::iterator_category,
			decltype(tuple_transform(detail::generic_dereference(),
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
	      = nullptr>  __host__ __device__
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
    template <class ITER_TUPLE_> __host__ __device__
    std::enable_if_t<std::is_convertible<ITER_TUPLE_, ITER_TUPLE>::value, bool>
		equal(const zip_iterator<ITER_TUPLE_>& iter) const
		{
		    return cuda::std::get<0>(iter.get_iterator_tuple())
			== cuda::std::get<0>(_iter_tuple);
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
    std::enable_if_t<std::is_convertible<ITER_TUPLE_, ITER_TUPLE>::value,
		     difference_type> __host__ __device__
		distance_to(const zip_iterator<ITER_TUPLE_>& iter) const
		{
		    return cuda::std::get<0>(iter.get_iterator_tuple())
			 - cuda::std::get<0>(_iter_tuple);
		}

  private:
    ITER_TUPLE	_iter_tuple;
};

template <class... ITERS> __host__ __device__ __forceinline__
zip_iterator<cuda::std::tuple<ITERS...> >
make_zip_iterator(const std::tuple<ITERS...>& iter_tuple)
{
    return {iter_tuple};
}

template <class ITER> __host__ __device__ __forceinline__ ITER
make_zip_iterator(const ITER& iter)
{
    return iter;
}

template <class ITER, class... ITERS> __host__ __device__ __forceinline__ auto
make_zip_iterator(const ITER& iter, const ITERS&... iters)
{
    return TU::cu::make_zip_iterator(cuda::std::make_tuple(iter, iters...));
}
#endif 
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
 *  Define global functions below in the namespace cuda::std
 *  so that they can be invoked in a unqualied manner using ADL.
 */
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
