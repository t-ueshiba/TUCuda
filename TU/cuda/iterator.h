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
#include "TU/cuda/tuple.h"

namespace TU
{
namespace cuda
{
/************************************************************************
*  map_iterator<FUNC, ITER>						*
************************************************************************/
template <class FUNC, class ITER>
class map_iterator
    : public thrust::iterator_adaptor<
	map_iterator<FUNC, ITER>,
	ITER,
	std::decay_t<
	    decltype(apply(std::declval<FUNC>(),
			   std::declval<typename std::iterator_traits<ITER>
						    ::reference>()))>,
	thrust::use_default,
	thrust::use_default,
	decltype(apply(std::declval<FUNC>(),
		       std::declval<typename std::iterator_traits<ITER>
						::reference>()))>
{
  private:
    using ref	= decltype(
			apply(std::declval<FUNC>(),
			      std::declval<typename std::iterator_traits<ITER>
						       ::reference>()));
    using super	= thrust::iterator_adaptor<map_iterator,
					   ITER,
					   std::decay_t<ref>,
					   thrust::use_default,
					   thrust::use_default,
					   ref>;
    friend	class thrust::iterator_core_access;

  public:
    using	typename super::reference;
	
  public:
    __host__ __device__
		map_iterator(FUNC func, const ITER& iter)
		    :super(iter), _func(func)				{}
	
  private:
    __host__ __device__
    reference	dereference()	const	{ return apply(_func, *super::base()); }
	
  private:
    FUNC	_func;	//!< 演算子
};
    
template <class FUNC, class ITER>
__host__ __device__ inline map_iterator<FUNC, ITER>
make_map_iterator(FUNC func, const ITER& iter)
{
    return {func, iter};
}

template <class FUNC, class ITER0, class ITER1, class... ITERS>
__host__ __device__
inline map_iterator<
	FUNC, thrust::zip_iterator<thrust::tuple<ITER0, ITER1, ITERS...> > >
make_map_iterator(FUNC func,
		  const ITER0& iter0, const ITER1& iter1, const ITERS&... iters)
{
    return {func,
	    thrust::make_zip_iterator(
		thrust::make_tuple(iter0, iter1, iters...))};
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

  public:
    __host__ __device__
		assignment_iterator(const FUNC& func, const ITER& iter)
		    :super(iter), _func(func)				{}

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
make_assignment_iterator(const FUNC& func, const ITER& iter)
{
    return {func, iter};
}

template <class FUNC, class ITER0, class ITER1, class... ITERS>
__host__ __device__
inline assignment_iterator<
	FUNC, thrust::zip_iterator<thrust::tuple<ITER0, ITER1, ITERS...> > >
make_assignment_iterator(const FUNC& func, const ITER0& iter0,
			 const ITER1& iter1, const ITERS&... iters)
{
    return {func,
	    thrust::make_zip_iterator(
		thrust::make_tuple(iter0, iter1, iters...))};
}

/************************************************************************
*  TU::cuda::stride(const ITER&)					*
************************************************************************/
template <class ITER_TUPLE> __host__ __device__ inline auto
stride(const thrust::zip_iterator<ITER_TUPLE>& iter)
{
    return tuple_transform([](const auto& iter){ return stride(iter); },
			   iter.get_iterator_tuple());
}

template <class FUNC, class ITER> __host__ __device__ inline auto
stride(const map_iterator<FUNC, ITER>& iter)
    -> decltype(stride(iter.base()))
{
    return stride(iter.base());
}

template <class FUNC, class ITER> __host__ __device__ inline auto
stride(const assignment_iterator<FUNC, ITER>& iter)
    -> decltype(stride(iter.base()))
{
    return stride(iter.base());
}

/************************************************************************
*  advance_stride(ITER&, STRIDE)					*
************************************************************************/
template <class ITER> __host__ __device__ inline auto
advance_stride(ITER& iter, const iterator_stride<ITER>& stride)
    -> void_t<decltype(iter += stride)>
{
    iter += stride;
}

/*
  thrust::zip_iterator<> の stride を thrust::tuple<> にするためには，
  本関数を有効化する．
*/
  /*
template <class ITER> __host__ __device__ inline auto
advance_stride(ITER& iter, ptrdiff_t stride)
    -> void_t<decltype(iter += stride)>
{
    iter += stride;
}

template <class FUNC, class ITER> __host__ __device__ inline auto
advance_stride(map_iterator<FUNC, ITER>& iter, ptrdiff_t stride)
    -> void_t<decltype(iter += stride)>
{
    iter += stride;
}

template <class FUNC, class ITER> __host__ __device__ inline auto
advance_stride(assignment_iterator<FUNC, ITER>& iter, ptrdiff_t stride)
    -> void_t<decltype(iter += stride)>
{
    iter += stride;
}
  */
template <class ITER_TUPLE, class HEAD, class TAIL>
__host__ __device__ inline auto
advance_stride(thrust::zip_iterator<ITER_TUPLE>& iter,
	       const thrust::detail::cons<HEAD, TAIL>& stride)
{
    using tuple_t = std::decay_t<decltype(iter.get_iterator_tuple())>;
    
    tuple_for_each([](auto&& x, const auto& y)
		   { using TU::advance_stride; advance_stride(x, y); },
		   const_cast<tuple_t&>(iter.get_iterator_tuple()), stride);
}

template <class ITER, class HEAD, class TAIL> __host__ __device__ inline auto
advance_stride(ITER& iter, const thrust::detail::cons<HEAD, TAIL>& stride)
    -> TU::void_t<decltype(iter.base())>
{
    using base_t = std::decay_t<decltype(iter.base())>;
    
    advance_stride(const_cast<base_t&>(iter.base()), stride);
}

}	// namespace cuda
}	// namespace TU
#endif	// !TU_CUDA_ITERATOR_H
