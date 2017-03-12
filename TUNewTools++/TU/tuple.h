/*!
  \file		tuple.h
  \brief	std::tupleの用途拡張のためのユティリティ
*/
#ifndef __TU_TUPLE_H
#define __TU_TUPLE_H

#include <tuple>
#include <functional>		// std::ref()
#include <iostream>
#include "TU/iterator.h"

namespace TU
{
/************************************************************************
*  predicate is_tuple<T>, is_range_tuple<T>				*
************************************************************************/
namespace detail
{
  template <class... T>
  std::tuple<T...>	check_tuple(std::tuple<T...>)			;
  void			check_tuple(...)				;

  template <class... T, size_t... IDX> auto
  check_range_tuple(std::tuple<T...> x, std::index_sequence<IDX...>)
      -> decltype(std::make_tuple(std::begin(std::get<IDX>(x))...))	;
  template <class... T> auto
  check_range_tuple(std::tuple<T...> x)
      -> decltype(check_range_tuple(x, std::index_sequence_for<T...>()));
  void
  check_range_tuple(...)						;
}	// namespace detail
    
template <class T>
using tuple_t	     = decltype(detail::check_tuple(std::declval<T>()));
template <class T>
using is_tuple	     = std::integral_constant<
			   bool, !std::is_void<tuple_t<T> >::value>;
template <class T>
using range_tuple_t  = decltype(detail::check_range_tuple(std::declval<T>()));
template <class T>
using is_range_tuple = std::integral_constant<
			   bool, !std::is_void<range_tuple_t<T> >::value>;

/************************************************************************
*  tuple_for_each(TUPLE, UNARY_FUNC)					*
************************************************************************/
namespace detail
{
  template <class TUPLE, class UNARY_FUNC> inline void
  tuple_for_each(TUPLE&, UNARY_FUNC, std::index_sequence<>)
  {
  }
  template <class TUPLE, class UNARY_FUNC, size_t I, size_t... IDX> inline void
  tuple_for_each(TUPLE& x, UNARY_FUNC f, std::index_sequence<I, IDX...>)
  {
      f(std::get<I>(x));
      tuple_for_each(x, f, std::index_sequence<IDX...>());
  }
}	// namespace detail
    
template <class TUPLE, class UNARY_FUNC>
inline std::enable_if_t<is_tuple<TUPLE>::value>
tuple_for_each(TUPLE&& x, UNARY_FUNC f)
{
    detail::tuple_for_each(
	x, f,
	std::make_index_sequence<std::tuple_size<tuple_t<TUPLE> >::value>());
}

/************************************************************************
*  tuple_for_each(TUPLE0, TUPLE1, BINARY_FUNC)				*
************************************************************************/
namespace detail
{
  template <class TUPLE0, class TUPLE1, class BINARY_FUNC> inline void
  tuple_for_each(TUPLE0&, TUPLE1&, BINARY_FUNC, std::index_sequence<>)
  {
  }
  template <class TUPLE0, class TUPLE1,
	    class BINARY_FUNC, size_t I, size_t... IDX> inline void
  tuple_for_each(TUPLE0& x, TUPLE1& y, BINARY_FUNC f,
		 std::index_sequence<I, IDX...>)
  {
      f(std::get<I>(x), std::get<I>(y));
      tuple_for_each(x, y, f, std::index_sequence<IDX...>());
  }
}	// namespace detail

template <class TUPLE0, class TUPLE1, class BINARY_FUNC>
inline std::enable_if_t<is_tuple<TUPLE0>::value && is_tuple<TUPLE1>::value>
tuple_for_each(TUPLE0&& x, TUPLE1&& y, BINARY_FUNC f)
{
    detail::tuple_for_each(
	x, y, f,
	std::make_index_sequence<std::tuple_size<tuple_t<TUPLE0> >::value>());
}

/************************************************************************
*  tuple_transform(TUPLE, UNARY_FUNC)					*
************************************************************************/
namespace detail
{
  template <class TUPLE, class UNARY_FUNC, size_t... IDX>
  inline auto
  tuple_transform(TUPLE& x, UNARY_FUNC f, std::index_sequence<IDX...>)
  {
      return std::make_tuple(f(std::get<IDX>(x))...);
  }
}	// namespace detail
    
template <class TUPLE, class UNARY_FUNC,
	  std::enable_if_t<is_tuple<TUPLE>::value>* = nullptr>
inline auto
tuple_transform(TUPLE&& x, UNARY_FUNC f)
{
    return detail::tuple_transform(
		x, f,
		std::make_index_sequence<std::tuple_size<tuple_t<TUPLE> >
		   ::value>());
}

/************************************************************************
*  tuple_transform(TUPLE0, TUPLE1, BINARY_FUNC)				*
************************************************************************/
namespace detail
{
  template <class TUPLE0, class TUPLE1, class BINARY_FUNC, size_t... IDX>
  inline auto
  tuple_transform(TUPLE0& x, TUPLE1& y, BINARY_FUNC f,
		  std::index_sequence<IDX...>)
  {
      return std::make_tuple(f(std::get<IDX>(x), std::get<IDX>(y))...);
  }
}	// namespace detail
    
template <class TUPLE0, class TUPLE1, class BINARY_FUNC,
	  std::enable_if_t<is_tuple<TUPLE0>::value &&
			   is_tuple<TUPLE1>::value>* = nullptr>
inline auto
tuple_transform(TUPLE0&& x, TUPLE1&& y, BINARY_FUNC f)
{
    return detail::tuple_transform(
		x, y, f,
		std::make_index_sequence<std::tuple_size<tuple_t<TUPLE0> >
		   ::value>());
}

/************************************************************************
*  tuple_transform(TUPLE0, TUPLE1, TUPLE2, TRINARY_FUNC)		*
************************************************************************/
namespace detail
{
  template <class TUPLE0, class TUPLE1, class TUPLE2, class TRINARY_FUNC,
	    size_t... IDX> inline auto
  tuple_transform(TUPLE0& x, TUPLE1& y, TUPLE2& z, TRINARY_FUNC f,
		  std::index_sequence<IDX...>)
  {
      return std::make_tuple(f(std::get<IDX>(x), std::get<IDX>(y),
			       std::get<IDX>(z))...);
  }
}	// namespace detail
    
template <class TUPLE0, class TUPLE1, class TUPLE2, class TRINARY_FUNC,
	  std::enable_if_t<is_tuple<TUPLE0>::value &&
			   is_tuple<TUPLE1>::value &&
			   is_tuple<TUPLE2>::value>* = nullptr>
inline auto
tuple_transform(TUPLE0&& x, TUPLE1&& y, TUPLE2&& z, TRINARY_FUNC f)
{
    return detail::tuple_transform(
		x, y, z, f,
		std::make_index_sequence<std::tuple_size<tuple_t<TUPLE0> >
		   ::value>());
}

/************************************************************************
*  Selection								*
************************************************************************/
template <class X, class Y> inline auto
select(bool s, const X& x, const Y& y)
{
    return (s ? x : y);
}
    
template <class... S, class... X, class... Y> inline auto
select(const std::tuple<S...>& s,
       const std::tuple<X...>& x, const std::tuple<Y...>& y)
{
    return tuple_transform(s, x, y,
			   [](const auto& t, const auto& u, const auto& v)
			   { return select(t, u, v); });
}

template <class... S, class X, class... Y> inline auto
select(const std::tuple<S...>& s, const X& x, const std::tuple<Y...>& y)
{
    return tuple_transform(s, y, [&x](const auto& t, const auto& v)
				 { return select(t, x, v); });
}

template <class... S, class... X, class Y> inline auto
select(const std::tuple<S...>& s, const std::tuple<X...>& x, const Y& y)
{
    return tuple_transform(s, x, [&y](const auto& t, const auto& u)
				 { return select(t, u, y); });
}

/************************************************************************
*  class unarizer<FUNC>							*
************************************************************************/
//! 引数をtupleにまとめることによって多変数関数を1変数関数に変換
template <class FUNC>
class unarizer
{
  public:
    using functor_type = FUNC;

  public:
    unarizer(FUNC func=FUNC())	:_func(func)		{}

    template <class TUPLE_,
	      std::enable_if_t<is_tuple<TUPLE_>::value>* = nullptr>
    auto	operator ()(const TUPLE_& arg) const
		{
		    return exec(arg, std::make_index_sequence<
					 std::tuple_size<TUPLE_>::value>());
		}

    const FUNC&	functor()			const	{return _func;}

  private:
    template <class TUPLE_, size_t... IDX_>
    auto	exec(const TUPLE_& arg, std::index_sequence<IDX_...>) const
		{
		    return _func(std::get<IDX_>(arg)...);
		}

  private:
    const FUNC	_func;
};

template <class FUNC> inline unarizer<FUNC>
make_unarizer(FUNC func)
{
    return {func};
}
    
/************************************************************************
*  class zip_iterator<ITER_TUPLE>					*
************************************************************************/
namespace detail
{
  struct generic_dereference
  {
      template <class ITER_>
      std::conditional_t<
	  std::is_reference<
	      typename std::iterator_traits<ITER_>::reference>::value,
	  std::reference_wrapper<
	      typename std::iterator_traits<ITER_>::value_type>,
	  typename std::iterator_traits<ITER_>::reference>
      operator ()(ITER_ iter)			const	{ return *iter; }
  };
}	// namespace detail
    
template <class ITER_TUPLE>
class zip_iterator
    : public boost::iterator_facade<
	  zip_iterator<ITER_TUPLE>,
	  decltype(tuple_transform(std::declval<ITER_TUPLE>(),
				   detail::generic_dereference())),
	  typename std::iterator_traits<
	      typename std::tuple_element<0, ITER_TUPLE>::type>
			  ::iterator_category,
	  decltype(tuple_transform(std::declval<ITER_TUPLE>(),
				   detail::generic_dereference()))>
{
  private:
    using super = boost::iterator_facade<
		      zip_iterator,
		      decltype(
			  tuple_transform(std::declval<ITER_TUPLE>(),
					  detail::generic_dereference())),
		      typename std::iterator_traits<
			  typename std::tuple_element<0, ITER_TUPLE>::type>
				      ::iterator_category,
		      decltype(
			  tuple_transform(std::declval<ITER_TUPLE>(),
					  detail::generic_dereference()))>;
    
  public:
    using		typename super::reference;
    using		typename super::difference_type;
    
    friend class	boost::iterator_core_access;

  public:
    zip_iterator(ITER_TUPLE iter_tuple)
	:_iter_tuple(iter_tuple)		{}

    const ITER_TUPLE&
		get_iterator_tuple()	const	{ return _iter_tuple; }
    
  private:
    reference	dereference() const
		{
		    return tuple_transform(_iter_tuple,
					   detail::generic_dereference());
		}
    bool	equal(const zip_iterator& iter) const
		{
		    return std::get<0>(iter.get_iterator_tuple())
			== std::get<0>(_iter_tuple);
		}
    void	increment()
		{
		    tuple_for_each(_iter_tuple, [](auto& x){ ++x; });
		}
    void	decrement()
		{
		    tuple_for_each(_iter_tuple, [](auto& x){ --x; });
		}
    void	advance(difference_type n)
		{
		    tuple_for_each(_iter_tuple, [n](auto& x){ x += n; });
		}
    difference_type
		distance_to(const zip_iterator& iter) const
		{
		    return std::get<0>(iter.get_iterator_tuple())
			 - std::get<0>(_iter_tuple);
		}

  private:
    ITER_TUPLE	_iter_tuple;
};

template <class ITER_TUPLE> inline zip_iterator<ITER_TUPLE>
make_zip_iterator(ITER_TUPLE iter_tuple)
{
    return {iter_tuple};
}

}	// namespace TU

/*
 *  argument dependent lookup が働くために，std::tuple<...>を引数とする
 *  operator overloadされた関数は namespace std 中に定義しなければならない．
 */
namespace std
{
/************************************************************************
*  std::[begin|end|rbegin|rend](std::tuple<T...>)			*
************************************************************************/
/*
 *  icpc-17.0.2 のバグ回避のため，lambda関数ではなくgenericな関数オブジェクトを
 *  用いて実装
 */ 
namespace detail
{
  struct generic_begin
  {
      template <class T_>
      auto	operator ()(T_&& x)	const	{ return std::begin(x); }
  };
  struct generic_end
  {
      template <class T_>
      auto	operator ()(T_&& x)	const	{ return std::end(x); }
  };
  struct generic_rbegin
  {
      template <class T_>
      auto	operator ()(T_&& x)	const	{ return std::rbegin(x); }
  };
  struct generic_rend
  {
      template <class T_>
      auto	operator ()(T_&& x)	const	{ return std::rend(x); }
  };
}	// namespace detail
    
template <class TUPLE, enable_if_t<TU::is_range_tuple<TUPLE>::value>* = nullptr>
inline auto
begin(TUPLE&& t)
{
    return TU::make_zip_iterator(TU::tuple_transform(
				     t, detail::generic_begin()));
  //return TU::make_zip_iterator(TU::tuple_transform(
  //				     t, [](auto&& x){ return begin(x); }));
}

template <class TUPLE, enable_if_t<TU::is_range_tuple<TUPLE>::value>* = nullptr>
inline auto
end(TUPLE&& t)
{
    return TU::make_zip_iterator(TU::tuple_transform(
				     t, detail::generic_end()));
  //return TU::make_zip_iterator(TU::tuple_transform(
  //				     t, [](auto&& x){ return end(x); }));
}
    
template <class TUPLE, enable_if_t<TU::is_range_tuple<TUPLE>::value>* = nullptr>
inline auto
rbegin(TUPLE&& t)
{
    return TU::make_zip_iterator(TU::tuple_transform(
				     t, detail::generic_rbegin()));
  //return TU::make_zip_iterator(TU::tuple_transform(
  //				     t, [](auto&& x){ return rbegin(x); }));
}

template <class TUPLE, enable_if_t<TU::is_range_tuple<TUPLE>::value>* = nullptr>
inline auto
rend(TUPLE&& t)
{
    return TU::make_zip_iterator(TU::tuple_transform(
				     t, detail::generic_rend()));
  //return TU::make_zip_iterator(TU::tuple_transform(
  //				     t, [](auto&& x){ return rend(x); }));
}
    
template <class... T> inline size_t
size(const tuple<T...>& t)
{
    return size(get<0>(t));
}
    
/************************************************************************
*  Arithmetic operators							*
************************************************************************/
template <class... T> inline auto
operator -(const tuple<T...>& t)
{
    return TU::tuple_transform(t, [](const auto& x){ return -x; });
}

template <class... L, class... R> inline auto
operator +(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x + y; });
}

template <class... L, class... R> inline auto
operator -(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x - y; });
}

template <class... L, class... R> inline auto
operator *(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x * y; });
}

template <class... L, class T> inline auto
operator *(const tuple<L...>& l, const T& c)
{
    return TU::tuple_transform(l, [&c](const auto& x){ return x * c; });
}

template <class T, class... R> inline auto
operator *(const T& c, const tuple<R...>& r)
{
    return TU::tuple_transform(r, [&c](const auto& x){ return c * x; });
}

template <class... L, class... R> inline auto
operator /(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x / y; });
}

template <class... L, class T> inline auto
operator /(const tuple<L...>& l, const T& c)
{
    return TU::tuple_transform(l, [&c](const auto& x){ return x / c; });
}

template <class... L, class... R> inline auto
operator %(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x % y; });
}

template <class... L, class T> inline auto
operator %(const tuple<L...>& l, const T& c)
{
    return TU::tuple_transform(l, [&c](const auto& x){ return x % c; });
}

template <class L, class... R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator +=(L&& l, const tuple<R...>& r)
{
    TU::tuple_for_each(l, r, [](auto& x, const auto& y){ x += y; });
    return l;
}

template <class L, class... R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator -=(L&& l, const tuple<R...>& r)
{
    TU::tuple_for_each(l, r, [](auto& x, const auto& y){ x -= y; });
    return l;
}

template <class L, class... R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator *=(L&& l, const tuple<R...>& r)
{
    TU::tuple_for_each(l, r, [](auto& x, const auto& y){ x *= y; });
    return l;
}

template <class L, class T> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator *=(L&& l, const T& c)
{
    TU::tuple_for_each(l, [&c](auto& x){ x *= c; });
    return l;
}

template <class L, class... R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator /=(L&& l, const tuple<R...>& r)
{
    TU::tuple_for_each(l, r, [](auto& x, const auto& y){ x /= y; });
    return l;
}

template <class L, class T> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator /=(L&& l, const T& c)
{
    TU::tuple_for_each(l, [&c](auto& x){ x /= c; });
    return l;
}

template <class L, class... R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator %=(L&& l, const tuple<R...>& r)
{
    TU::tuple_for_each(l, r, [](auto& x, const auto& y){ x %= y; });
    return l;
}

template <class L, class T> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator %=(L&& l, const T& c)
{
    TU::tuple_for_each(l, [&c](auto& x){ x %= c; });
    return l;
}

/************************************************************************
*  Bit operators							*
************************************************************************/
template <class... L, class... R> inline auto
operator &(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x & y; });
}
    
template <class... L, class... R> inline auto
operator |(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x | y; });
}
    
template <class... L, class... R> inline auto
operator ^(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x ^ y; });
}
    
template <class L, class... R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator &=(L&& l, const tuple<R...>& r)
{
    TU::tuple_for_each(l, r, [](auto& x, const auto& y){ x &= y; });
    return l;
}

template <class L, class... R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator |=(L&& l, const tuple<R...>& r)
{
    TU::tuple_for_each(l, r, [](auto& x, const auto& y){ x |= y; });
    return l;
}

template <class L, class... R> inline enable_if_t<TU::is_tuple<L>::value, L&>
operator ^=(L&& l, const tuple<R...>& r)
{
    TU::tuple_for_each(l, r, [](auto& x, const auto& y){ x ^= y; });
    return l;
}

/************************************************************************
*  Logical operators							*
************************************************************************/
template <class... T> inline auto
operator !(const tuple<T...>& t)
{
    return TU::tuple_transform(t, [](const auto& x){ return !x; });
}
    
template <class... L, class... R> inline auto
operator &&(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x && y; });
}
    
template <class... L, class... R> inline auto
operator ||(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x || y; });
}
    
/************************************************************************
*  Relational operators							*
************************************************************************/
template <class... L, class... R> inline auto
operator ==(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x == y; });
}
    
template <class... L, class... R> inline auto
operator !=(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x != y; });
}
    
template <class... L, class... R> inline auto
operator <(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x < y; });
}
    
template <class... L, class... R> inline auto
operator >(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x > y; });
}
    
template <class... L, class... R> inline auto
operator <=(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x <= y; });
}
    
template <class... L, class... R> inline auto
operator >=(const tuple<L...>& l, const tuple<R...>& r)
{
    return TU::tuple_transform(l, r, [](const auto& x, const auto& y)
				     { return x >= y; });
}

/************************************************************************
*  I/O functions							*
************************************************************************/
template <class... T> inline ostream&
operator <<(ostream& out, const tuple<T...>& t)
{
    out << '(';
    TU::tuple_for_each(t, [&out](const auto& x){ out << ' ' << x; });
    out << ')';

    return out;
}

}	// namespace std
#endif	// !__TU_TUPLE_H