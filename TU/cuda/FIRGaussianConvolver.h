/*
 *  $Id$
 */
/*!
  \file		FIRGaussianConvolver.h
  \brief	Gauss核による畳み込みに関連するクラスの定義と実装
*/ 
#ifndef TU_CUDA_FIRGAUSSIANCONVOLVER_H
#define TU_CUDA_FIRGAUSSIANCONVOLVER_H

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

    template <class IN, class OUT> void	smooth(IN in, IN ie, OUT out)	;
    template <class IN, class OUT> void	diffH (IN in, IN ie, OUT out)	;
    template <class IN, class OUT> void	diffV (IN in, IN ie, OUT out)	;
    template <class IN, class OUT> void	diffHH(IN in, IN ie, OUT out)	;
    template <class IN, class OUT> void	diffHV(IN in, IN ie, OUT out)	;
    template <class IN, class OUT> void	diffVV(IN in, IN ie, OUT out)	;
    
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
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class IN, class OUT> inline void
FIRGaussianConvolver2::smooth(IN in, IN ie, OUT out)
{
    FIRFilter2::initialize(_lobe0, _lobe0).convolve(in, ie, out);
}
    
//! Gauss核による横方向1階微分(DOG)
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class IN, class OUT> inline void
FIRGaussianConvolver2::diffH(IN in, IN ie, OUT out)
{
    FIRFilter2::initialize(_lobe1, _lobe0).convolve(in, ie, out);
}
    
//! Gauss核による縦方向1階微分(DOG)
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class IN, class OUT> inline void
FIRGaussianConvolver2::diffV(IN in, IN ie, OUT out)
{
    FIRFilter2::initialize(_lobe0, _lobe1).convolve(in, ie, out);
}
    
//! Gauss核による横方向2階微分
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class IN, class OUT> inline void
FIRGaussianConvolver2::diffHH(IN in, IN ie, OUT out)
{
    FIRFilter2::initialize(_lobe2, _lobe0).convolve(in, ie, out);
}
    
//! Gauss核による縦横両方向2階微分
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class IN, class OUT> inline void
FIRGaussianConvolver2::diffHV(IN in, IN ie, OUT out)
{
    FIRFilter2::initialize(_lobe1, _lobe1).convolve(in, ie, out);
}
    
//! Gauss核による縦方向2階微分
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class IN, class OUT> inline void
FIRGaussianConvolver2::diffVV(IN in, IN ie, OUT out)
{
    FIRFilter2::initialize(_lobe0, _lobe2).convolve(in, ie, out);
}
    
}
}
#endif	// !TU_CUDA_FIRGAUSSIANCONVOLVER_H
