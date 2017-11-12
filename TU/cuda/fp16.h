/*
 *  $Id$
 */
/*!
  \file		fp16.h
  \brief	半精度浮動小数点に燗する各種アルゴリズムの定義と実装

  本ヘッダを使用する場合，nvccに -arch=sm_53 以上を，g++に -mf16c を指定する．
*/ 
#ifndef TU_CUDA_FP16_H
#define TU_CUDA_FP16_H

#include <cuda_fp16.h>		// for __half
#include <emmintrin.h>		// for _cvtss_sh() and _cvtsh_ss()
#include <thrust/device_ptr.h>
#include "TU/Array++.h"

namespace TU
{
/************************************************************************
*  struct to_half							*
************************************************************************/
//! 指定された型から半精度浮動小数点数へ変換する関数オブジェクト
struct to_half
{
    template <class T>
    __half	operator ()(T x) const
		{
		    const auto	y = _cvtss_sh(x, 0);
		    return *(reinterpret_cast<const __half*>(&y));
		}
};

/************************************************************************
*  struct from_half<T>							*
************************************************************************/
//! 半精度浮動小数点数から指定された型へ変換する関数オブジェクト
/*!
  \param T	変換先の型
*/
template <class T>
struct from_half
{
    T	operator ()(__half x) const
	{
	    return T(_cvtsh_ss(*reinterpret_cast<const unsigned short*>(&x)));
	}
};

}	// namespace TU

namespace thrust
{
/************************************************************************
*  algorithms overloaded for thrust::device_ptr<__half>			*
************************************************************************/
template <size_t N, class S> inline void
copy(const S* p, size_t n, device_ptr<__half> q)
{
    copy_n(TU::make_map_iterator(TU::to_half(), p), (N ? N : n), q);
}

template <size_t N, class T> inline void
copy(device_ptr<const __half> p, size_t n, T* q)
{
#if 0
    copy_n(p, (N ? N : n),
	   TU::make_assignment_iterator(q, TU::from_half<T>()));
#else
    TU::Array<__half, N>	tmp(n);
    copy_n(p, (N ? N : n), tmp.begin());
    std::copy_n(tmp.cbegin(), (N ? N : n),
		TU::make_assignment_iterator(q, TU::from_half<T>()));
#endif
}

}	// namespace thrust
#endif	// !TU_CUDA_FP16_H
