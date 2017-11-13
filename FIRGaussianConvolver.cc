/*
 *  $Id$
 */
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
    
    const size_t	sizMax  = FIRFilter2_traits::LobeSizeMax;
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
