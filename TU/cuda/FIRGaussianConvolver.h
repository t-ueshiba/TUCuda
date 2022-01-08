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
  \file		FIRGaussianConvolver.h
  \brief	Gauss核による畳み込みに関連するクラスの定義と実装
*/
#pragma once

#include "TU/cuda/FIRFilter.h"

namespace TU
{
namespace cuda
{
namespace detail
{
  size_t	lobeSize(const float lobe[], bool even)			;
}
/************************************************************************
*  class FIRGaussianConvolver2<T>					*
************************************************************************/
//! CUDAを用いてGauss核により2次元配列畳み込みを行うクラス
template <class T=float>
class FIRGaussianConvolver2 : public FIRFilter2<T>
{
  private:
    using super	= FIRFilter2<T>;

  public:
    FIRGaussianConvolver2(float sigma=1.0)				;

    float			sigma()				const	;
    FIRGaussianConvolver2&	initialize(float sigma)			;

    template <class IN, class OUT>
    void	smooth(IN in, IN ie, OUT out, bool shift=false)		;
    template <class IN, class OUT>
    void	diffH( IN in, IN ie, OUT out, bool shift=false)		;
    template <class IN, class OUT>
    void	diffV( IN in, IN ie, OUT out, bool shift=false)		;
    template <class IN, class OUT>
    void	diffHH(IN in, IN ie, OUT out, bool shift=false)		;
    template <class IN, class OUT>
    void	diffHV(IN in, IN ie, OUT out, bool shift=false)		;
    template <class IN, class OUT>
    void	diffVV(IN in, IN ie, OUT out, bool shift=false)		;

  private:
    float		_sigma;
    TU::Array<float>	_lobe0;		//!< スムージングのためのローブ
    TU::Array<float>	_lobe1;		//!< 1階微分のためのローブ
    TU::Array<float>	_lobe2;		//!< 2階微分のためのローブ
};

//! Gauss核を生成する
/*!
  \param sigma	Gauss核のスケール
*/
template <class T> inline
FIRGaussianConvolver2<T>::FIRGaussianConvolver2(float sigma)
    :_sigma(sigma)
{
    initialize(_sigma);
}

template <class T> inline float
FIRGaussianConvolver2<T>::sigma() const
{
    return _sigma;
}

//! Gauss核を初期化する
/*!
  \param sigma	Gauss核のスケール
  \return	このGauss核
*/
template <class T>
FIRGaussianConvolver2<T>&
FIRGaussianConvolver2<T>::initialize(float sigma)
{
    using namespace	std;

    _sigma = sigma;

  // 0/1/2階微分のためのローブを計算する．
    const size_t	sizMax = super::LobeSizeMax;
    float		lobe0[sizMax], lobe1[sizMax], lobe2[sizMax];
    for (size_t i = 0; i < sizMax; ++i)
    {
	float	dx = float(i) / _sigma, dxdx = dx*dx;

	lobe0[i] = exp(-0.5f * dxdx);
	lobe1[i] = -dx * lobe0[i];
	lobe2[i] = (dxdx - 1.0f) * lobe0[i];
    }

  // 0階微分用のローブを正規化して格納する．
    _lobe0.resize(detail::lobeSize(lobe0, true));
    float	sum = lobe0[0];
    for (size_t i = 1; i < _lobe0.size(); ++i)
	sum += (2.0f * lobe0[i]);
    for (size_t i = 0; i < _lobe0.size(); ++i)
	_lobe0[i] = lobe0[_lobe0.size() - 1 - i] / abs(sum);

  // 1階微分用のローブを正規化して格納する．
    _lobe1.resize(detail::lobeSize(lobe1, false));
    sum = 0.0f;
    for (size_t i = 0; i < _lobe1.size(); ++i)
	sum += (2.0f * i * lobe1[i]);
    for (size_t i = 0; i < _lobe1.size(); ++i)
	_lobe1[i] = lobe1[_lobe1.size() - i] / abs(sum);

  // 2階微分用のローブを正規化して格納する．
    _lobe2.resize(detail::lobeSize(lobe2, true));
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

//! Gauss核によるスムーシング
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline void
FIRGaussianConvolver2<T>::smooth(IN in, IN ie, OUT out, bool shift)
{
    super::initialize(_lobe0, _lobe0).convolve(in, ie, out, shift);
}

//! Gauss核による横方向1階微分(DOG)
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline void
FIRGaussianConvolver2<T>::diffH(IN in, IN ie, OUT out, bool shift)
{
    super::initialize(_lobe1, _lobe0).convolve(in, ie, out, shift);
}

//! Gauss核による縦方向1階微分(DOG)
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline void
FIRGaussianConvolver2<T>::diffV(IN in, IN ie, OUT out, bool shift)
{
    super::initialize(_lobe0, _lobe1).convolve(in, ie, out, shift);
}

//! Gauss核による横方向2階微分
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline void
FIRGaussianConvolver2<T>::diffHH(IN in, IN ie, OUT out, bool shift)
{
    super::initialize(_lobe2, _lobe0).convolve(in, ie, out, shift);
}

//! Gauss核による縦横両方向2階微分
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline void
FIRGaussianConvolver2<T>::diffHV(IN in, IN ie, OUT out, bool shift)
{
    super::initialize(_lobe1, _lobe1).convolve(in, ie, out, shift);
}

//! Gauss核による縦方向2階微分
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class T> template <class IN, class OUT> inline void
FIRGaussianConvolver2<T>::diffVV(IN in, IN ie, OUT out, bool shift)
{
    super::initialize(_lobe0, _lobe2).convolve(in, ie, out, shift);
}

}
}
