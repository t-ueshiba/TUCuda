/*
 *  $Id: CudaFilter.h,v 1.2 2011-04-14 08:39:34 ueshiba Exp $
 */
#ifndef __TUCudaFilter_h
#define __TUCudaFilter_h

#include "TU/CudaArray++.h"

namespace TU
{
/************************************************************************
*  class CudaFilter2							*
************************************************************************/
//! CUDAによるseparableな2次元フィルタを表すクラス
class CudaFilter2
{
  public:
    enum		{LOBE_SIZE_MAX = 17};

  public:
    CudaFilter2()							;
    
    CudaFilter2&	initialize(const Array<float>& lobeH,
				   const Array<float>& lobeV)		;
    template <class S, class T>
    const CudaFilter2&	convolve(const CudaArray2<S>& in,
				       CudaArray2<T>& out)	const	;
    template <class S, class T>
    const CudaFilter2&	convolveH(const CudaArray2<S>& in,
					CudaArray2<T>& out)	const	;
    template <class S, class T>
    const CudaFilter2&	convolveV(const CudaArray2<S>& in,
					CudaArray2<T>& out)	const	;

  private:
    cudaDeviceProp		_prop;		//!< デバイスの特性
    u_int			_lobeSizeH;	//!< 水平方向フィルタのローブ長
    u_int			_lobeSizeV;	//!< 垂直方向フィルタのローブ長
    mutable CudaArray2<float>	_buf;		//!< 中間結果用のバッファ
};
    
}

#endif	// !__TUCudaFilter_h
