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
  template <class FUNC, class ITER>
  class assignment_proxy
  {
    private:
      template <class T_>
      static auto	check_func(ITER iter, const T_& val, FUNC func)
			    -> decltype(func(*iter, val), std::true_type());
      template <class T_>
      static auto	check_func(ITER iter, const T_& val, FUNC func)
			    -> decltype(*iter = func(val), std::false_type());
      template <class T_>
      using is_binary_func	= decltype(check_func(std::declval<ITER>(),
						      std::declval<T_>(),
						      std::declval<FUNC>()));
      
    public:
      __host__ __device__
      assignment_proxy(const ITER& iter, const FUNC& func)
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
      const ITER&	_iter;
      const FUNC&	_func;
  };
}	// namespace detail
    
//! operator *()を左辺値として使うときに，この左辺値と右辺値に指定された関数を適用するための反復子
/*!
  \param FUNC	変換を行う関数オブジェクトの型
  \param ITER	変換結果の代入先を指す反復子
*/
template <class FUNC, class ITER>
class assignment_iterator
    : public thrust::iterator_adaptor<assignment_iterator<FUNC, ITER>,
				      ITER,
				      thrust::use_default,
				      thrust::use_default,
				      thrust::use_default,
				      detail::assignment_proxy<FUNC, ITER> >
{
  private:
    using super	= thrust::iterator_adaptor<
			assignment_iterator,
			ITER,
			thrust::use_default,
			thrust::use_default,
			thrust::use_default,
			detail::assignment_proxy<FUNC, ITER> >;
    
    friend	class thrust::iterator_core_access;
    
  public:
    using	typename super::reference;
    using	typename super::difference_type;

  public:
    __host__ __device__
    assignment_iterator(const ITER& iter, const FUNC& func=FUNC())
	:super(iter), _func(func)	{}

    const auto&	functor()	const	{ return _func; }

  private:
    __host__ __device__
    reference	dereference()	const	{ return {super::base(), _func}; }
    
  private:
    FUNC 	_func;	// 代入を可能にするためconstは付けない
};
#endif	// __NVCC__
    
template <class FUNC, class ITER>
__host__ __device__ inline assignment_iterator<FUNC, ITER>
make_assignment_iterator(const ITER& iter, const FUNC& func=FUNC())
{
    return {iter, func};
}

}	// namespace cuda
}	// namespace TU
#endif	// !TU_CUDA_ITERATOR_H
