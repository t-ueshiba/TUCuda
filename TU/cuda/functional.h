/*
 *  $Id$
 */
/*!
  \file		functional.h
  \brief	CUDAデバイス上で実行される関数オブジェクトの定義と実装
*/
#pragma once

#include <cmath>
#include <thrust/functional.h>
#include "TU/cuda/vec.h"

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

/************************************************************************
*  undistortion operators						*
************************************************************************/
template <class ITER_K, class ITER_D> void
set_intrinsic_parameters(ITER_K K, ITER_D d)				;

#if defined(__NVCC__)
namespace device
{
  /*
   *  Static __constant__ variables
   */
  template <class T> __constant__ static vec<T, 2>	_flen[1];
  template <class T> __constant__ static vec<T, 2>	_uv0[1];
  template <class T> __constant__ static T		_d[4];

}	// namespace device

template <class T>
struct undistort
{
    template <class ITER_K, class ITER_D>
    undistort(ITER_K K, ITER_D d)
    {
	vec<T, 2>	flen, uv0;
	flen.x = *K;
	std::advance(K, 2);
	uv0.x = *K;
	std::advance(K, 2);
	flen.y = *K;
	++K;
	uv0.y = *K;
	const T	dd[] = {*d, *++d, *++d, *++d};

	cudaMemcpyToSymbol(device::_flen<T>, &flen, sizeof(flen), 0,
			   cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(device::_uv0<T>, &uv0, sizeof(uv0), 0,
			   cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(device::_d<T>,  dd, sizeof(dd), 0,
			   cudaMemcpyHostToDevice);
    }
    
    __device__ vec<T, 2>
    operator ()(T u, T v) const
    {
	static constexpr T	MAX_ERR  = 0.001*0.001;
	static constexpr int	MAX_ITER = 5;

	const vec<T, 2>	uv{u, v};
	auto		xy  = (uv - device::_uv0<T>[0])/device::_flen<T>[0];
	const auto	xy0 = xy;

      // compensate distortion iteratively
	for (int n = 0; n < MAX_ITER; ++n)
	{
	    const auto	r2 = cuda::square(xy);
	    const auto	k  = T(1) + (device::_d<T>[0] + device::_d<T>[1]*r2)*r2;
	    if (k < T(0))
		break;

	    const auto	a = T(2)*xy.x*xy.y;
	    vec<T, 2>	delta{device::_d<T>[2]*a +
			      device::_d<T>[3]*(r2 + T(2)*xy.x*xy.x),
			      device::_d<T>[2]*(r2 + T(2)*xy.y*xy.y) +
			      device::_d<T>[3]*a};
	    const auto	uv_proj = device::_flen<T>[0]*(k*xy + delta)
				+ device::_uv0<T>[0];

	    if (cuda::square(uv_proj - uv) < MAX_ERR)
		break;

	    xy = (xy0 - delta)/k;	// compensate lens distortion
	}

	return xy;
    }

    __device__ vec<T, 3>
    operator ()(T u, T v, T d) const
    {
	const auto	xy = (*this)(u, v);
	return {d*xy.x, d*xy.y, d};
    }
};

template <class T, class ITER_K, class ITER_D> void
set_intrinsic_parameters(ITER_K K, ITER_D d)
{
    vec<T, 2>	flen, uv0;
    flen.x = *K;
    std::advance(K, 2);
    uv0.x = *K;
    std::advance(K, 2);
    flen.y = *K;
    ++K;
    uv0.y = *K;
    const T	dd[] = {*d, *++d, *++d, *++d};

    cudaMemcpyToSymbol(device::_flen<T>, &flen, sizeof(flen), 0,
    		       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(device::_uv0<T>, &uv0, sizeof(uv0), 0,
		       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(device::_d<T>,  dd, sizeof(dd), 0,
		       cudaMemcpyHostToDevice);
}
#endif    
}	// namespace cuda
}	// namespace TU
