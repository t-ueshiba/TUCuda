/*
 *  $Id$
 */
/*!
  \file		CudaFilter.h
  \brief	フィルタの定義と実装
*/ 
#ifndef __TU_CUDAFILTER_H
#define __TU_CUDAFILTER_H

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

  private:
    cudaDeviceProp		_prop;		//!< デバイスの特性
    size_t			_lobeSizeH;	//!< 水平方向フィルタのローブ長
    size_t			_lobeSizeV;	//!< 垂直方向フィルタのローブ長
    mutable CudaArray2<float>	_buf;		//!< 中間結果用のバッファ
};
    
}
#endif	// !__TU_CUDAFILTER_H
