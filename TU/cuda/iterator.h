/*
 *  $Id$
 */
/*!
  \file		iterator.h
  \brief	半精度浮動小数点に燗する各種アルゴリズムの定義と実装
*/ 
#ifndef TU_CUDA_ITERATOR_H
#define TU_CUDA_ITERATOR_H

#include <type_traits>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/zip_iterator.h>

namespace TU
{
namespace cuda
{
/************************************************************************
*  map_iterator<FUNC, ITER...>						*
************************************************************************/
template <class FUNC, class... ITER>
class map_iterator
    : public thrust::iterator_adaptor<
	map_iterator<FUNC, ITER...>,
	thrust::zip_iterator<thrust::tuple<ITER...> >,
	std::decay_t<
	    std::result_of_t<
		FUNC(typename std::iterator_traits<ITER>::reference...)> >,
	thrust::use_default,
	thrust::use_default,
	std::result_of_t<
	    FUNC(typename std::iterator_traits<ITER>::reference...)> >
{
  private:
    using ref	= std::result_of_t<
		      FUNC(typename std::iterator_traits<ITER>::reference...)>;
    using super	= thrust::iterator_adaptor<map_iterator,
					   thrust::zip_iterator<
					       thrust::tuple<ITER...> >,
					   std::decay_t<ref>,
					   thrust::use_default,
					   thrust::use_default,
					   ref>;
    friend	class thrust::iterator_core_access;

  public:
    using	typename super::difference_type;
    using	typename super::reference;
	
  public:
    __host__ __device__
		map_iterator(FUNC func, const ITER&... iter)
		    :super(thrust::tuple<ITER...>(iter...)), _func(func)
		{
		}
	
  private:
    __host__ __device__
    reference	dereference() const
		{
		    return dereference(
				std::make_index_sequence<sizeof...(ITER)>());
		}
    template <size_t... IDX_> __host__ __device__
    reference	dereference(std::index_sequence<IDX_...>) const
		{
		    return _func(thrust::get<IDX_>(*super::base())...);
		}
	
  private:
    FUNC	_func;	//!< 演算子
};
    
template <class FUNC, class... ITER>
__host__ __device__ inline map_iterator<FUNC, ITER...>
make_map_iterator(FUNC func, const ITER&... iter)
{
    return {func, iter...};
}

/************************************************************************
*  class assignment_iterator<FUNC, ITER>				*
************************************************************************/
#if defined(__NVCC__)
namespace detail
{
  template <class FUNC, class ITER, class... ITERS>
  class assignment_proxy
  {
    public:
      using iterator	= std::conditional_t<
				sizeof...(ITERS),
				thrust::zip_iterator<
				    thrust::tuple<ITER, ITERS...> >,
				ITER>;
      
    private:
      template <class T_>
      static auto	check_func(iterator iter, const T_& val, FUNC func)
			    -> decltype(func(*iter, val), std::true_type());
      template <class T_>
      static auto	check_func(iterator iter, const T_& val, FUNC func)
			    -> decltype(*iter = func(val), std::false_type());
      template <class T_>
      using is_binary_func	= decltype(check_func(std::declval<iterator>(),
						      std::declval<T_>(),
						      std::declval<FUNC>()));
      
    public:
      __host__ __device__
      assignment_proxy(const iterator& iter, const FUNC& func)
	  :_iter(iter), _func(func)					{}

      template <class T_> __host__ __device__
      std::enable_if_t<is_binary_func<T_>::value, assignment_proxy&>
			operator =(T_&& val)
			{
			    _func(*_iter, std::forward<T_>(val));
			    return *this;
			}
      template <class T_> __host__ __device__
      std::enable_if_t<!is_binary_func<T_>::value, assignment_proxy&>
			operator =(T_&& val)
			{
			    *_iter  = _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_> __host__ __device__
      assignment_proxy&	operator +=(T_&& val)
			{
			    *_iter += _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_> __host__ __device__
      assignment_proxy&	operator -=(T_&& val)
			{
			    *_iter -= _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_> __host__ __device__
      assignment_proxy&	operator *=(T_&& val)
			{
			    *_iter *= _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_> __host__ __device__
      assignment_proxy&	operator /=(T_&& val)
			{
			    *_iter /= _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_> __host__ __device__
      assignment_proxy&	operator &=(T_&& val)
			{
			    *_iter &= _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_> __host__ __device__
      assignment_proxy&	operator |=(T_&& val)
			{
			    *_iter |= _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_> __host__ __device__
      assignment_proxy&	operator ^=(T_&& val)
			{
			    *_iter ^= _func(std::forward<T_>(val));
			    return *this;
			}

    private:
      const iterator&	_iter;
      const FUNC&	_func;
  };
}	// namespace detail
    
//! operator *()を左辺値として使うときに，この左辺値と右辺値に指定された関数を適用するための反復子
/*!
  \param FUNC	変換を行う関数オブジェクトの型
  \param ITER	変換結果の代入先を指す反復子
*/
template <class FUNC, class... ITER>
class assignment_iterator
    : public thrust::iterator_adaptor<
		assignment_iterator<FUNC, ITER...>,
		typename detail::assignment_proxy<FUNC, ITER...>::iterator,
		thrust::use_default,
		thrust::use_default,
		thrust::use_default,
		detail::assignment_proxy<FUNC, ITER...> >
{
  private:
    using super	= thrust::iterator_adaptor<
			assignment_iterator,
			typename
			    detail::assignment_proxy<FUNC, ITER...>::iterator,
			thrust::use_default,
			thrust::use_default,
			thrust::use_default,
			detail::assignment_proxy<FUNC, ITER...> >;
    friend	class thrust::iterator_core_access;
    
  public:
    using	typename super::reference;

  public:
    __host__ __device__
    assignment_iterator(const FUNC& func, const ITER&... iter)
	:assignment_iterator(
	    std::integral_constant<bool, (sizeof...(ITER) > 1)>(),
	    func, iter...)					{}

    const auto&	functor()	const	{ return _func; }

  private:
    __host__ __device__
    assignment_iterator(std::true_type,  const FUNC& func, const ITER&... iter)
	:super(thrust::make_tuple(iter...)), _func(func)	{}
    __host__ __device__
    assignment_iterator(std::false_type, const FUNC& func, const ITER&... iter)
	:super(iter...), _func(func)				{}

    __host__ __device__
    reference	dereference()	const	{ return {super::base(), _func}; }
    
  private:
    FUNC 	_func;	// 代入を可能にするためconstは付けない
};
#endif	// __NVCC__
    
template <class FUNC, class... ITER>
__host__ __device__ inline assignment_iterator<FUNC, ITER...>
make_assignment_iterator(const FUNC& func, const ITER&... iter)
{
    return {func, iter...};
}

}	// namespace cuda

template <class HEAD, class TAIL> __host__ __device__ inline auto
make_zip_iterator(const thrust::detail::cons<HEAD, TAIL>& iter_tuple)
{
    return thrust::make_zip_iterator(iter_tuple);
}

}	// namespace TU
#endif	// !TU_CUDA_ITERATOR_H
