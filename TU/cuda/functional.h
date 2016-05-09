/*
 *  $Id$
 */
/*!
  \file		functional.h
  \brief	CUDAデバイス上で実行される関数オブジェクトの定義と実装
*/
#ifndef __TU_CUDA_FUNCTIONAL_H
#define __TU_CUDA_FUNCTIONAL_H

#include <cmath>
#include <thrust/functional.h>

namespace TU
{
namespace cuda
{
/************************************************************************
*  3x3 operators							*
************************************************************************/
//! 横方向1階微分オペレータを表す関数オブジェクト
template <class T>
struct diffH3x3
{
    typedef T	result_type;
    
    template <class ITER> __host__ __device__ T
    operator ()(ITER p, ITER c, ITER n) const
    {
	return T(0.5)*(c[2] - c[0]);
    }
};
    
//! 縦方向1階微分オペレータを表す関数オブジェクト
template <class T>
struct diffV3x3
{
    typedef T	result_type;
    
    template <class ITER> __host__ __device__ T
    operator ()(ITER p, ITER c, ITER n) const
    {
	return T(0.5)*(n[1] - p[1]);
    }
};
    
//! 横方向2階微分オペレータを表す関数オブジェクト
template <class T>
struct diffHH3x3
{
    typedef T	result_type;
    
    template <class ITER> __host__ __device__ T
    operator ()(ITER p, ITER c, ITER n) const
    {
	return c[0] - T(2)*c[1] + c[2];
    }
};
    
//! 縦方向2階微分オペレータを表す関数オブジェクト
template <class T>
struct diffVV3x3
{
    typedef T	result_type;
    
    template <class ITER> __host__ __device__ T
    operator ()(ITER p, ITER c, ITER n) const
    {
	return p[1] - T(2)*c[1] + n[1];
    }
};
    
//! 縦横両方向2階微分オペレータを表す関数オブジェクト
template <class T>
struct diffHV3x3
{
    typedef T	result_type;
    
    template <class ITER> __host__ __device__ T
    operator ()(ITER p, ITER c, ITER n) const
    {
	return T(0.25)*(p[0] - p[2] - n[0] + n[2]);
    }
};
    
//! 横方向1階微分Sobelオペレータを表す関数オブジェクト
template <class T>
struct sobelH3x3
{
    typedef T	result_type;
    
    template <class ITER> __host__ __device__ T
    operator ()(ITER p, ITER c, ITER n) const
    {
	return T(0.125)*(p[2] - p[0] + n[2] - n[0]) + T(0.250)*(c[2] - c[0]);
    }
};
    
//! 縦方向1階微分Sobelオペレータを表す関数オブジェクト
template <class T>
struct sobelV3x3
{
    typedef T	result_type;
    
    template <class ITER> __host__ __device__ T
    operator ()(ITER p, ITER c, ITER n) const
    {
	return T(0.125)*(n[0] - p[0] + n[2] - p[2]) + T(0.250)*(n[1] - p[1]);
    }
};
    
//! 1階微分Sobelオペレータの縦横両方向出力の絶対値の和を表す関数オブジェクト
template <class T>
struct sobelAbs3x3
{
    typedef T	result_type;
    
    template <class ITER> __host__ __device__ T
    operator ()(ITER p, ITER c, ITER n) const
    {
	return std::abs(sobelH3x3<T>()(p, c, n))
	     + std::abs(sobelV3x3<T>()(p, c, n));
    }
};
    
//! ラプラシアンオペレータを表す関数オブジェクト
template <class T>
struct laplacian3x3
{
    typedef T	result_type;
    
    template <class ITER> __host__ __device__ T
    operator ()(ITER p, ITER c, ITER n) const
    {
	return c[0] + c[2] + p[1] + n[1] - T(4)*c[1];
    }
};
    
//! ヘッセ行列式オペレータを表す関数オブジェクト
template <class T>
struct det3x3
{
    typedef T	result_type;
    
    template <class ITER> __host__ __device__ T
    operator ()(ITER p, ITER c, ITER n) const
    {
	const T	dxy = diffHV3x3<T>()(p, c, n);
	
	return diffHH3x3<T>()(p, c, n) * diffVV3x3<T>()(p, c, n) - dxy * dxy;
    }
};

//! 極大点検出オペレータを表す関数オブジェクト
template <class T>
class maximal3x3
{
  public:
    typedef T	result_type;

    __host__ __device__
    maximal3x3(T nonMaximal=0)	:_nonMaximal(nonMaximal)	{}
    
    template <class ITER> __host__ __device__ T
    operator ()(ITER p, ITER c, ITER n) const
    {
	return ((c[1] > p[0]) && (c[1] > p[1]) && (c[1] > p[2]) &&
		(c[1] > c[0])		       && (c[1] > c[2]) &&
		(c[1] > n[0]) && (c[1] > n[1]) && (c[1] > n[2]) ?
		c[1] : _nonMaximal);
    }

  private:
    const T	_nonMaximal;
};

//! 極小点検出オペレータを表す関数オブジェクト
template <class T>
class minimal3x3
{
  public:
    typedef T	result_type;

    __host__ __device__
    minimal3x3(T nonMinimal=0)	:_nonMinimal(nonMinimal)	{}
    
    template <class ITER> __host__ __device__ T
    operator ()(ITER p, ITER c, ITER n) const
    {
	return ((c[1] < p[0]) && (c[1] < p[1]) && (c[1] < p[2]) &&
		(c[1] < c[0])		       && (c[1] < c[2]) &&
		(c[1] < n[0]) && (c[1] < n[1]) && (c[1] < n[2]) ?
		c[1] : _nonMinimal);
    }

  private:
    const T	_nonMinimal;
};

//! 2つの値の閾値付き差を表す関数オブジェクト
template <class T>
struct diff
{
    __host__ __device__
    diff(T thresh)	:_thresh(thresh)			{}

    __host__ __device__ T
    operator ()(T x, T y) const
    {
	return thrust::minimum<T>()((x > y ? x - y : y - x), _thresh);
    }
    
  private:
    const T	_thresh;
};
    
}	// namespace cuda
}	// namespace TU
#endif	// !__TU_CUDA_FUNCTIONAL_H
