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
  \file		FIRGaussianConvolver.cc
  \brief	Gauss核による畳み込みに関連するクラスの実装
*/
#include "TU/cuda/FIRGaussianConvolver.h"
#include <cmath>

namespace TU
{
namespace cuda
{
namespace detail
{
/************************************************************************
*  static functions							*
************************************************************************/
size_t
lobeSize(const float lobe[], bool even)
{
    using namespace	std;
    
    const size_t	sizMax  = FIRFilter2<>::LobeSizeMax;
    const float	epsilon = 0.01;			// 打ち切りのしきい値の比率

  // 打ち切りのしきい値を求める．
    float	th = 0;
    for (size_t i = sizMax; i-- > 0; )
	if (abs(lobe[i]) >= th)
	    th = abs(lobe[i]);
    th *= epsilon;

  // しきい値を越える最大のローブ長を求める．
    size_t	siz;
    for (siz = sizMax; siz-- > 0; )		// ローブ長を縮める
	if (abs(lobe[siz]) > th)		// しきい値を越えるまで
	{
	    ++siz;
	    break;
	}

    if (even)
    {
	if (siz <= 2)
	    return 3;		// 2^1 + 1
	else if (siz <= 5)
	    return 5;		// 2^2 + 1
	else if (siz <= 9)
	    return 9;		// 2^3 + 1
	else
	    return 17;		// 2^4 + 1
    }
    else
    {
	if (siz <= 1)
	    return 2;		// 2^1
	else if (siz <= 4)
	    return 4;		// 2^2
	else if (siz <= 8)
	    return 8;		// 2^3
	else
	    return 16;		// 2^4
    }
}
    
}	// namespace detail
}	// namespace cuda
}	// namespace TU
