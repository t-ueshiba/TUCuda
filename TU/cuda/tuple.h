/*!
  \file		tuple.h
  \author	Toshio UESHIBA
  \brief	thrust::tupleの用途拡張のためのユティリティ
*/
#ifndef TU_CUDA_TUPLE_H
#define TU_CUDA_TUPLE_H

#include "TU/iterator.h"
#include "TU/cuda/iterator.h"

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
}	// namespace cuda
}	// namepsace TU

namespace thrust
{
/************************************************************************
*  predicate: is_cons<T>						*
************************************************************************/
namespace detail
{
  template <class HEAD, class TAIL>
  std::true_type	check_cons(cons<HEAD, TAIL>)			;
  std::false_type	check_cons(...)					;
}	// namespace detail

//! 与えられた型が thrust::tuple 又はそれに変換可能であるか判定する
/*!
  \param T	判定対象となる型
*/ 
template <class T>
using is_cons = decltype(detail::check_cons(std::declval<T>()));

/************************************************************************
*  predicate: any_cons<ARGS...>					*
************************************************************************/
//! 少なくとも1つのテンプレート引数が thrust::tuple 又はそれに変換可能であるか判定する
/*!
  \param ARGS...	判定対象となる型の並び
*/
template <class... ARGS>
struct any_cons : std::false_type					{};
template <class ARG, class... ARGS>
struct any_cons<ARG, ARGS...>
    : std::integral_constant<bool, (is_cons<ARG>::value ||
				    any_cons<ARGS...>::value)>		{};
    
/************************************************************************
*  predicate: any_null<ARGS...>					*
************************************************************************/
//! 少なくとも1つのテンプレート引数が thrust::null_type 又はそれに変換可能であるか判定する
/*!
  \param ARGS...	判定対象となる型の並び
*/
template <class... ARGS>
struct any_null : std::false_type					{};
template <class ARG, class... ARGS>
struct any_null<ARG, ARGS...>
    : std::integral_constant<
	bool, (std::is_convertible<ARG, null_type>::value ||
	       any_null<ARGS...>::value)>				{};
    
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
  struct replace_element<tuple<S...>, T>
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
__host__ __device__ inline std::enable_if_t<any_null<TUPLES...>::value>
tuple_for_each(FUNC, TUPLES&&...)
{
}

template <class FUNC, class... TUPLES>
__host__ __device__ inline std::enable_if_t<any_cons<TUPLES...>::value>
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
      return cons<HEAD, TAIL>(std::forward<HEAD>(head),
			      std::forward<TAIL>(tail));
  }

  template <class FUNC, class TUPLE> inline auto
  tuple_transform(std::index_sequence<>, FUNC, TUPLE&&)
  {
      return null_type();
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
inline std::enable_if_t<!any_cons<TUPLES...>::value, null_type>
tuple_transform(FUNC, TUPLES&&...)
{
    return null_type();
}
template <class FUNC, class... TUPLES,
	  std::enable_if_t<any_cons<TUPLES...>::value>* = nullptr>
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

template <class L, class R, std::enable_if_t<any_cons<L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator +(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y){ return x + y; },
			   l, r);
}

template <class L, class R, std::enable_if_t<any_cons<L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator -(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y){ return x - y; },
			   l, r);
}

template <class L, class R, std::enable_if_t<any_cons<L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator *(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y){ return x * y; },
			   l, r);
}

template <class L, class R, std::enable_if_t<any_cons<L, R>::value>* = nullptr>
__host__ __device__ inline auto
operator /(const L& l, const R& r)
{
    return tuple_transform([](const auto& x, const auto& y){ return x / y; },
			   l, r);
}

template <class L, class R, std::enable_if_t<any_cons<L, R>::value>* = nullptr>
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
  print(std::ostream& out, null_type)
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
  print(std::ostream& out, const cons<HEAD, TAIL>& x)
  {
      return print(print(out << ' ', get_head(x)), get_tail(x));
  }

  template <class HEAD, class TAIL> inline std::ostream&
  operator <<(std::ostream& out, const cons<HEAD, TAIL>& x)
  {
      return print(out << '(', x) << ')';
  }
}
    
/************************************************************************
*  begin/end functions for tuples					*
************************************************************************/
template <class TUPLE, std::enable_if_t<is_cons<TUPLE>::value>* = nullptr>
inline auto
begin(TUPLE&& t)
{
    return make_zip_iterator(tuple_transform(
				 [](auto&& x)
				 { using TU::begin; return begin(x); },
				 std::forward<TUPLE>(t)));
}

template <class TUPLE, std::enable_if_t<is_cons<TUPLE>::value>* = nullptr>
__host__ __device__ inline auto
end(TUPLE&& t)
{
    return make_zip_iterator(tuple_transform(
				 [](auto&& x)
				 { using TU::end; return end(x); },
				 std::forward<TUPLE>(t)));
}

template <class TUPLE, std::enable_if_t<is_cons<TUPLE>::value>* = nullptr>
__host__ __device__ inline auto
rbegin(TUPLE&& t)
{
    return std::make_reverse_iterator(end(std::forward<TUPLE>(t)));
}

template <class TUPLE, std::enable_if_t<is_cons<TUPLE>::value>* = nullptr>
__host__ __device__ inline auto
rend(TUPLE&& t)
{
    return std::make_reverse_iterator(begin(std::forward<TUPLE>(t)));
}

template <class TUPLE, std::enable_if_t<is_cons<TUPLE>::value>* = nullptr>
__host__ __device__ inline auto
cbegin(const TUPLE& t)
{
    return begin(t);
}

template <class TUPLE, std::enable_if_t<is_cons<TUPLE>::value>* = nullptr>
__host__ __device__ inline auto
cend(const TUPLE& t)
{
    return end(t);
}

template <class TUPLE, std::enable_if_t<is_cons<TUPLE>::value>* = nullptr>
__host__ __device__ inline auto
crbegin(const TUPLE& t)
{
    return rbegin(t);
}

template <class TUPLE, std::enable_if_t<is_cons<TUPLE>::value>* = nullptr>
__host__ __device__ inline auto
crend(const TUPLE& t)
{
    return rend(t);
}

template <class TUPLE>
__host__ __device__ inline std::enable_if_t<is_cons<TUPLE>::value, size_t>
size(const TUPLE& t)
{
    return TU::size(get<0>(t));
}

}	// namespace thrust
#endif	// !TU_CUDA_TUPLE_H
