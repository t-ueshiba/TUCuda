/*
 *  $Id$
 */
/*!
  \file		FIRGaussianConvolver.h
  \brief	Gauss核による畳み込みに関連するクラスの定義と実装
*/ 
#ifndef __TU_CUDA_FIRGAUSSIANCONVOLVER_H
#define __TU_CUDA_FIRGAUSSIANCONVOLVER_H

#include "TU/cuda/FIRFilter.h"

namespace TU
{
namespace cuda
{
/************************************************************************
*  class FIRGaussianConvolver2						*
************************************************************************/
//! CUDAを用いてGauss核により2次元配列畳み込みを行うクラス
class FIRGaussianConvolver2 : public FIRFilter2
{
  public:
    FIRGaussianConvolver2(float sigma=1.0)				;

    FIRGaussianConvolver2&	initialize(float sigma)			;

    template <class S, class T> FIRGaussianConvolver2&
	smooth(const Array2<S>& in, Array2<T>& out)			;
    template <class S, class T> FIRGaussianConvolver2&
	diffH(const Array2<S>& in, Array2<T>& out)			;
    template <class S, class T> FIRGaussianConvolver2&
	diffV(const Array2<S>& in, Array2<T>& out)			;
    template <class S, class T> FIRGaussianConvolver2&
	diffHH(const Array2<S>& in, Array2<T>& out)			;
    template <class S, class T> FIRGaussianConvolver2&
	diffHV(const Array2<S>& in, Array2<T>& out)			;
    template <class S, class T> FIRGaussianConvolver2&
	diffVV(const Array2<S>& in, Array2<T>& out)			;
    
  private:
    TU::Array<float>	_lobe0;		//!< スムージングのためのローブ
    TU::Array<float>	_lobe1;		//!< 1階微分のためのローブ
    TU::Array<float>	_lobe2;		//!< 2階微分のためのローブ
};
    
//! Gauss核を生成する
/*!
  \param sigma	Gauss核のスケール
*/
inline
FIRGaussianConvolver2::FIRGaussianConvolver2(float sigma)
{
    initialize(sigma);
}

//! Gauss核によるスムーシング
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このGauss核自身
*/
template <class S, class T> inline FIRGaussianConvolver2&
FIRGaussianConvolver2::smooth(const Array2<S>& in, Array2<T>& out)
{
    FIRFilter2::initialize(_lobe0, _lobe0).convolve(in, out);

    return *this;
}
    
//! Gauss核による横方向1階微分(DOG)
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このGauss核自身
*/
template <class S, class T> inline FIRGaussianConvolver2&
FIRGaussianConvolver2::diffH(const Array2<S>& in, Array2<T>& out)
{
    FIRFilter2::initialize(_lobe1, _lobe0).convolve(in, out);

    return *this;
}
    
//! Gauss核による縦方向1階微分(DOG)
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このGauss核自身
*/
template <class S, class T> inline FIRGaussianConvolver2&
FIRGaussianConvolver2::diffV(const Array2<S>& in, Array2<T>& out)
{
    FIRFilter2::initialize(_lobe0, _lobe1).convolve(in, out);

    return *this;
}
    
//! Gauss核による横方向2階微分
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このGauss核自身
*/
template <class S, class T> inline FIRGaussianConvolver2&
FIRGaussianConvolver2::diffHH(const Array2<S>& in, Array2<T>& out)
{
    FIRFilter2::initialize(_lobe2, _lobe0).convolve(in, out);

    return *this;
}
    
//! Gauss核による縦横両方向2階微分
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このGauss核自身
*/
template <class S, class T> inline FIRGaussianConvolver2&
FIRGaussianConvolver2::diffHV(const Array2<S>& in, Array2<T>& out)
{
    FIRFilter2::initialize(_lobe1, _lobe1).convolve(in, out);

    return *this;
}
    
//! Gauss核による縦方向2階微分
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このGauss核自身
*/
template <class S, class T> inline FIRGaussianConvolver2&
FIRGaussianConvolver2::diffVV(const Array2<S>& in, Array2<T>& out)
{
    FIRFilter2::initialize(_lobe0, _lobe2).convolve(in, out);

    return *this;
}
    
}
}
#endif	// !__TU_CUDA_FIRGAUSSIANCONVOLVER_H
