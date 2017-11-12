/*
 *  $Id$
 */
/*!
  \file		fp16.h
  \brief	半精度浮動小数点に燗する各種アルゴリズムの定義と実装

  本ヘッダを使用する場合，nvccに -arch=sm_53 以上を，g++に -mf16c を与える．
*/ 
#ifndef TU_CUDA_FP16_H
#define TU_CUDA_FP16_H

#include <cuda_fp16.h>
#include <emmintrin.h>
#include <boost/iterator/iterator_adaptor.hpp>
#include <thrust/device_ptr.h>

namespace TU
{
namespace cuda
{
/************************************************************************
*  class to_half<T>							*
************************************************************************/
//! 指定された型から半精度浮動小数点数へ変換する反復子クラス
/*!
  \param T	変換元の型
*/
template <class T>
class to_half : public boost::iterator_adaptor<to_half<T>,
					       const T*,
					       __half,
					       boost::use_default,
					       __half>
{
  private:
    using	super = boost::iterator_adaptor<to_half,
						const T*,
						__half,
						boost::use_default,
						__half>;
    friend	class boost::iterator_core_access;
    
  public:
    using	typename super::reference;
	
  public:
		to_half(const T* p)	:super(p)		{}

  private:
    reference	dereference() const
		{
		    const auto	tmp = _cvtss_sh(*super::base(), 0);
		    return *(reinterpret_cast<const __half*>(&tmp));
		}
};

/************************************************************************
*  class from_half<T>							*
************************************************************************/
namespace detail
{
  template <class T>
  class from_half_proxy
  {
    public:
      from_half_proxy(T* p)	:_p(p)				{}

      from_half_proxy&
      operator =(__half val)
      {
	  *_p = T(_cvtsh_ss(*reinterpret_cast<const unsigned short*>(&val)));
	  return *this;
      }

    private:
      T* const	_p;
  };
}	// namespace detail
    
//! 半精度浮動小数点数から指定された型へ変換する反復子クラス
/*!
  \param T	変換先の型
*/
template <class T>
class from_half
    : public boost::iterator_adaptor<from_half<T>,
				     T*,
				     T,
				     thrust::input_host_iterator_tag,
				     detail::from_half_proxy<T> >
{
  private:
    using	super = boost::iterator_adaptor<
				from_half,
				T*,
				T,
				thrust::input_host_iterator_tag,
				detail::from_half_proxy<T> >;
    friend	class boost::iterator_core_access;
    
  public:
    using	typename super::reference;

  public:
    from_half(T* p)	:super(p)	{}
    
  private:
    reference	dereference()	const	{ return {super::base()}; }
};

}	// namespace cuda
}	// namespace TU

namespace thrust
{
/************************************************************************
*  algorithms overloaded for thrust::device_ptr<__half>			*
************************************************************************/
template <size_t N, class S> inline void
copy(const S* p, size_t n, device_ptr<__half> q)
{
    copy_n(TU::cuda::to_half<S>(p), (N ? N : n), q);
}

template <size_t N, class T> inline void
copy(device_ptr<const __half> p, size_t n, T* q)
{
    copy_n(p, (N ? N : n), TU::cuda::from_half<T>(q));
}

}	// namespace thrust
#endif	// !TU_CUDA_FP16_H
