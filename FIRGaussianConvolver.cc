/*
 *  $Id$
 */
/*!
  \file		FIRGaussianConvolver.cc
*/
#include "TU/cuda/FIRGaussianConvolver.h"
#include <cmath>

namespace TU
{
namespace cuda
{
/************************************************************************
*  static functions							*
************************************************************************/
static size_t
lobeSize(const float lobe[], bool even)
{
    using namespace	std;
    
    const size_t	sizMax  = FIRFilter2::LobeSizeMax;
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
    
/************************************************************************
*  class FIRGaussianConvolver2						*
************************************************************************/
//! Gauss核を初期化する
/*!
  \param sigma	Gauss核のスケール
  \return	このGauss核
*/
FIRGaussianConvolver2&
FIRGaussianConvolver2::initialize(float sigma)
{
    using namespace	std;

  // 0/1/2階微分のためのローブを計算する．
    const size_t	sizMax = FIRFilter2::LobeSizeMax;
    float		lobe0[sizMax], lobe1[sizMax], lobe2[sizMax];
    for (size_t i = 0; i < sizMax; ++i)
    {
	float	dx = float(i) / sigma, dxdx = dx*dx;
	
	lobe0[i] = exp(-0.5f * dxdx);
	lobe1[i] = -dx * lobe0[i];
	lobe2[i] = (dxdx - 1.0f) * lobe0[i];
    }

  // 0階微分用のローブを正規化して格納する．
    _lobe0.resize(lobeSize(lobe0, true));
    float	sum = lobe0[0];
    for (size_t i = 1; i < _lobe0.size(); ++i)
	sum += (2.0f * lobe0[i]);
    for (size_t i = 0; i < _lobe0.size(); ++i)
	_lobe0[i] = lobe0[_lobe0.size() - 1 - i] / abs(sum);

  // 1階微分用のローブを正規化して格納する．
    _lobe1.resize(lobeSize(lobe1, false));
    sum = 0.0f;
    for (size_t i = 0; i < _lobe1.size(); ++i)
	sum += (2.0f * i * lobe1[i]);
    for (size_t i = 0; i < _lobe1.size(); ++i)
	_lobe1[i] = lobe1[_lobe1.size() - i] / abs(sum);

  // 2階微分用のローブを正規化して格納する．
    _lobe2.resize(lobeSize(lobe2, true));
    sum = 0.0f;
    for (size_t i = 1; i < _lobe2.size(); ++i)
	sum += (i * i * lobe2[i]);
    for (size_t i = 0; i < _lobe2.size(); ++i)
	_lobe2[i] = lobe2[_lobe2.size() - 1 - i] / abs(sum);

#ifdef _DEBUG
    cerr << "lobe0: " << _lobe0;
    cerr << "lobe1: " << _lobe1;
    cerr << "lobe2: " << _lobe2;
#endif
    
    return *this;
}
    
}
}
