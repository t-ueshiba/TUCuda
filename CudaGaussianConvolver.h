/*
 *  $Id: CudaGaussianConvolver.h,v 1.4 2011-04-20 08:15:07 ueshiba Exp $
 */
#include "TU/CudaFilter.h"

namespace TU
{
/************************************************************************
*  class CudaGaussianConvolver2						*
************************************************************************/
//! CUDAを用いてGauss核により2次元配列畳み込みを行うクラス
class CudaGaussianConvolver2 : public CudaFilter2
{
  public:
    CudaGaussianConvolver2(float sigma=1.0)				;

    CudaGaussianConvolver2&	initialize(float sigma)			;

    template <class S, class T> CudaGaussianConvolver2&
	smooth(const CudaArray2<S>& in, CudaArray2<T>& out)		;
    template <class S, class T> CudaGaussianConvolver2&
	diffH(const CudaArray2<S>& in, CudaArray2<T>& out)		;
    template <class S, class T> CudaGaussianConvolver2&
	diffV(const CudaArray2<S>& in, CudaArray2<T>& out)		;
    template <class S, class T> CudaGaussianConvolver2&
	diffHH(const CudaArray2<S>& in, CudaArray2<T>& out)		;
    template <class S, class T> CudaGaussianConvolver2&
	diffHV(const CudaArray2<S>& in, CudaArray2<T>& out)		;
    template <class S, class T> CudaGaussianConvolver2&
	diffVV(const CudaArray2<S>& in, CudaArray2<T>& out)		;
    
  private:
    Array<float>	_lobe0;		//!< スムージングのためのローブ
    Array<float>	_lobe1;		//!< 1階微分のためのローブ
    Array<float>	_lobe2;		//!< 2階微分のためのローブ
};
    
//! Gauss核を生成する
/*!
  \param sigma	Gauss核のスケール
*/
inline
CudaGaussianConvolver2::CudaGaussianConvolver2(float sigma)
{
    initialize(sigma);
}

//! Gauss核によるスムーシング
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このGauss核自身
*/
template <class S, class T> inline CudaGaussianConvolver2&
CudaGaussianConvolver2::smooth(const CudaArray2<S>& in, CudaArray2<T>& out)
{
    CudaFilter2::initialize(_lobe0, _lobe0).convolve(in, out);

    return *this;
}
    
//! Gauss核による横方向1階微分(DOG)
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このGauss核自身
*/
template <class S, class T> inline CudaGaussianConvolver2&
CudaGaussianConvolver2::diffH(const CudaArray2<S>& in, CudaArray2<T>& out)
{
    CudaFilter2::initialize(_lobe1, _lobe0).convolve(in, out);

    return *this;
}
    
//! Gauss核による縦方向1階微分(DOG)
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このGauss核自身
*/
template <class S, class T> inline CudaGaussianConvolver2&
CudaGaussianConvolver2::diffV(const CudaArray2<S>& in, CudaArray2<T>& out)
{
    CudaFilter2::initialize(_lobe0, _lobe1).convolve(in, out);

    return *this;
}
    
//! Gauss核による横方向2階微分
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このGauss核自身
*/
template <class S, class T> inline CudaGaussianConvolver2&
CudaGaussianConvolver2::diffHH(const CudaArray2<S>& in, CudaArray2<T>& out)
{
    CudaFilter2::initialize(_lobe2, _lobe0).convolve(in, out);

    return *this;
}
    
//! Gauss核による縦横両方向2階微分
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このGauss核自身
*/
template <class S, class T> inline CudaGaussianConvolver2&
CudaGaussianConvolver2::diffHV(const CudaArray2<S>& in, CudaArray2<T>& out)
{
    CudaFilter2::initialize(_lobe1, _lobe1).convolve(in, out);

    return *this;
}
    
//! Gauss核による縦方向2階微分
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このGauss核自身
*/
template <class S, class T> inline CudaGaussianConvolver2&
CudaGaussianConvolver2::diffVV(const CudaArray2<S>& in, CudaArray2<T>& out)
{
    CudaFilter2::initialize(_lobe0, _lobe2).convolve(in, out);

    return *this;
}
    
}
