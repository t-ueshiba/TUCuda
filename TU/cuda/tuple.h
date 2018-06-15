/*!
  \file		tuple.h
  \author	Toshio UESHIBA
  \brief	thrust::tupleの用途拡張のためのユティリティ
*/
#ifndef TU_CUDA_TUPLE_H
#define TU_CUDA_TUPLE_H

#include <thrust/iterator/zip_iterator.h>
#include "TU/tuple.h"			// for TU::is_tuple<TUPLE>

namespace TU
{
namespace cuda
{
/************************************************************************
*  alias: TU::cuda::tuple<T...>						*
************************************************************************/
namespace detail
{
  template <class... T>
  struct tuple_t;

  template <>
  struct tuple_t<>
  {
      using type = thrust::null_type;
  };
    
  template <class S, class... T>
  struct tuple_t<S, T...>
  {
      using type = thrust::detail::cons<S, typename tuple_t<T...>::type>;
  };
}	// namespace detail
    
template <class... T>
using tuple = typename detail::tuple_t<T...>::type;

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
*  type alias: replace_element<S, T>					*
************************************************************************/
namespace detail
{
  template <class S, class T>
  struct replace_element : std::conditional<std::is_void<T>::value, S, T>
  {
  };
  template <class... S, class T>
  struct replace_element<thrust::tuple<S...>, T>
  {
      using type = tuple<typename replace_element<S, T>::type...>;
  };
}	// namespace detail
    
//! 与えられた型がthrust::tupleならばその要素の型を，そうでなければ元の型自身を別の型で置き換える．
/*!
  \param S	要素型置換の対象となる型
  \param T	置換後の要素の型．voidならば置換しない．
*/
template <class S, class T>
using replace_element = typename detail::replace_element<S, T>::type;

/************************************************************************
*  tuple_for_each(TUPLES..., FUNC)				`	*
************************************************************************/
namespace detail
{
  template <class T, std::enable_if_t<!is_cons<T>::value>* = nullptr>
  __host__ __device__ inline decltype(auto)
  get_head(T&& x)
  {
      return std::forward<T>(x);
  }
  template <class T, std::enable_if_t<is_cons<T>::value>* = nullptr>
  __host__ __device__ inline decltype(auto)
  get_head(T&& x)
  {
      return x.get_head();
  }
    
  template <class T, std::enable_if_t<!is_cons<T>::value>* = nullptr>
  __host__ __device__ inline decltype(auto)
  get_tail(T&& x)
  {
      return std::forward<T>(x);
  }
  template <class T, std::enable_if_t<is_cons<T>::value>* = nullptr>
  __host__ __device__ inline decltype(auto)
  get_tail(T&& x)
  {
      return x.get_tail();
  }
}	// namespace detail

template <class FUNC, class... TUPLES>
__host__ __device__ inline std::enable_if_t<any<is_null, TUPLES...>::value>
tuple_for_each(FUNC, TUPLES&&...)
{
}

template <class FUNC, class... TUPLES>
__host__ __device__ inline std::enable_if_t<any<is_cons, TUPLES...>::value>
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
  template <class HEAD, class TAIL> __host__ __device__ inline auto
  make_cons(HEAD&& head, TAIL&& tail)
  {
      return thrust::detail::cons<HEAD, TAIL>(std::forward<HEAD>(head),
					      std::forward<TAIL>(tail));
  }

  template <class FUNC, class TUPLE> inline auto
  tuple_transform(std::index_sequence<>, FUNC, TUPLE&&)
  {
      return thrust::null_type();
  }
  template <class FUNC, class TUPLE, size_t I, size_t... IDX> inline auto
  tuple_transform(std::index_sequence<I, IDX...>, FUNC f, TUPLE&& x)
  {
      return make_cons(f(std::get<I>(std::forward<TUPLE>(x))),
		       tuple_transform(std::index_sequence<IDX...>(),
				       f, std::forward<TUPLE>(x)));
  }
}	// namespace detail

template <class FUNC, class... TUPLES> __host__ __device__
inline std::enable_if_t<!any<is_cons, TUPLES...>::value, thrust::null_type>
tuple_transform(FUNC, TUPLES&&...)
{
    return thrust::null_type();
}
template <class FUNC, class... TUPLES,
	  std::enable_if_t<any<is_cons, TUPLES...>::value>* = nullptr>
__host__ __device__ inline auto
tuple_transform(FUNC f, TUPLES&&... x)
{
    return detail::make_cons(f(detail::get_head(std::forward<TUPLES>(x))...),
			     tuple_transform(
				 f,
				 detail::get_tail(std::forward<TUPLES>(x))...));
}

template <class FUNC, class TUPLE,
	  std::enable_if_t<TU::is_tuple<TUPLE>::value>* = nullptr> inline auto
tuple_transform(FUNC f, TUPLE&& x)
{
    constexpr auto	tsize = std::tuple_size<std::decay_t<TUPLE> >::value;
    return detail::tuple_transform(std::make_index_sequence<tsize>(),
				   f, std::forward<TUPLE>(x));
}

/************************************************************************
*  Arithmetic operators							*
************************************************************************/
template <class T, std::enable_if_t<is_cons<T>::value>* = nullptr>
__host__ __device__ inline auto
operator -(const T& t)
{
    return tuple_transform([](const auto& x){ return -x; }, t);
}

template <class L, class R,
	  std::enable_if_t<any<is_cons, L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator +(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y){ return x + y; },
			   l, r);
}

template <class L, class R,
	  std::enable_if_t<any<is_cons, L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator -(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y){ return x - y; },
			   l, r);
}

template <class L, class R,
	  std::enable_if_t<any<is_cons, L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator *(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y){ return x * y; },
			   l, r);
}

template <class L, class R,
	  std::enable_if_t<any<is_cons, L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator /(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y){ return x / y; },
			   l, r);
}

template <class L, class R,
	  std::enable_if_t<any<is_cons, L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator %(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y){ return x % y; },
			   l, r);
}

template <class L, class R>
__host__ __device__ inline std::enable_if_t<is_cons<L>::value, L&>
operator +=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x += y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ inline std::enable_if_t<is_cons<L>::value, L&>
operator -=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x -= y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ inline std::enable_if_t<is_cons<L>::value, L&>
operator *=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x *= y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ inline std::enable_if_t<is_cons<L>::value, L&>
operator /=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x /= y; }, l, r);
    return l;
}

template <class L, class R>
__host__ __device__ inline std::enable_if_t<is_cons<L>::value, L&>
operator %=(L&& l, const R& r)
{
    tuple_for_each([](auto&& x, const auto& y){ x %= y; }, l, r);
    return l;
}

template <class T>
__host__ __device__ inline std::enable_if_t<is_cons<T>::value, T&>
operator ++(T&& t)
{
    tuple_for_each([](auto&& x){ ++x; }, t);
    return t;
}

template <class T>
__host__ __device__ inline std::enable_if_t<is_cons<T>::value, T&>
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
  inline std::enable_if_t<!is_cons<T>::value, std::ostream&>
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
  __host__ __device__ inline decltype(auto)
  apply(FUNC&& f, TUPLE&& t, std::index_sequence<IDX...>)
  {
      return f(thrust::get<IDX>(std::forward<TUPLE>(t))...);
  }
}
    
template <class FUNC, class TUPLE,
	  std::enable_if_t<is_cons<TUPLE>::value>* = nullptr>
__host__ __device__ inline decltype(auto)
apply(FUNC&& f, TUPLE&& t)
{
    return detail::apply(
		std::forward<FUNC>(f), std::forward<TUPLE>(t),
		std::make_index_sequence<
			thrust::tuple_size<std::decay_t<TUPLE> >::value>());
}
template <class FUNC, class T,
	  std::enable_if_t<!is_cons<T>::value>* = nullptr>
__host__ __device__ inline decltype(auto)
apply(FUNC&& f, T&& t)
{
    return f(std::forward<T>(t));
}
    
/************************************************************************
*  begin|end|rbegin|rend|size for tuples				*
************************************************************************/
template <class HEAD, class TAIL> inline auto
begin(thrust::detail::cons<HEAD, TAIL>& t)
{
    return thrust::make_zip_iterator(
		tuple_transform([](auto&& x)
				{ using std::begin; return begin(x); },
				t));
}

template <class HEAD, class TAIL> inline auto
end(thrust::detail::cons<HEAD, TAIL>& t)
{
    return thrust::make_zip_iterator(
		tuple_transform([](auto&& x)
				{ using std::end; return end(x); },
				t));
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
		tuple_transform([](auto&& x)
				{ using std::begin; return begin(x); },
				t));
}

template <class HEAD, class TAIL> inline auto
end(const thrust::detail::cons<HEAD, TAIL>& t)
{
    return thrust::make_zip_iterator(
		tuple_transform([](auto&& x)
				{ using std::end; return end(x); },
				t));
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
    return thrust::get<0>(t).size();
}

}	// namespace cuda
}	// namepsace TU

#endif	// !TU_CUDA_TUPLE_H
