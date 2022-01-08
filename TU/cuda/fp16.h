// Software License Agreement (BSD License)
//
// Copyright (c) 2021, National Institute of Advanced Industrial Science and Technology (AIST)
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//  * Neither the name of National Institute of Advanced Industrial
//    Science and Technology (AIST) nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Toshio Ueshiba
//
/*!
  \file		fp16.h
  \brief	半精度浮動小数点に燗する各種アルゴリズムの定義と実装

  本ヘッダを使用する場合，nvccに -arch=sm_53 以上を，g++に -mf16c を指定する．
*/
#pragma once

#include <cuda_fp16.h>		// for __half
#include <immintrin.h>		// for _cvtss_sh() and _cvtsh_ss()
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
		    const auto	y = _cvtss_sh(x, (_MM_FROUND_TO_NEAREST_INT |
						  _MM_FROUND_NO_EXC));
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
	   TU::make_assignment_iterator(TU::from_half<T>(), q));
#else
    TU::Array<__half, N>	tmp(n);
    copy_n(p, (N ? N : n), tmp.begin());
    std::copy_n(tmp.cbegin(), (N ? N : n),
		TU::make_assignment_iterator(TU::from_half<T>(), q));
#endif
}

}	// namespace thrust
