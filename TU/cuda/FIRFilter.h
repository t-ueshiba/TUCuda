/*
 *  $Id$
 */
/*!
  \file		FIRFilter.h
  \brief	フィルタの定義と実装
*/ 
#ifndef __TU_CUDA_FIRFILTER_H
#define __TU_CUDA_FIRFILTER_H

#include "TU/cuda/Array++.h"

namespace TU
{
namespace cuda
{
/************************************************************************
*  class FIRFilter2							*
************************************************************************/
//! CUDAによるseparableな2次元フィルタを表すクラス
class FIRFilter2
{
  public:
    enum		{LOBE_SIZE_MAX = 17};

  public:
    FIRFilter2()							;
    
    FIRFilter2&		initialize(const TU::Array<float>& lobeH,
				   const TU::Array<float>& lobeV)	;
    template <class S, class T>
    const FIRFilter2&	convolve(const Array2<S>& in,
				       Array2<T>& out)		const	;

  private:
    cudaDeviceProp		_prop;		//!< デバイスの特性
    size_t			_lobeSizeH;	//!< 水平方向フィルタのローブ長
    size_t			_lobeSizeV;	//!< 垂直方向フィルタのローブ長
    mutable Array2<float>	_buf;		//!< 中間結果用のバッファ
};

}
}
#endif	// !__TU_CUDA_FIRFILTER_H
