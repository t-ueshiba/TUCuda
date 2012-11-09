/*
 *  $Id$
 */
/*!
  \file		CudaUtility.h
  \brief	各種オペレータの定義と実装
*/ 
#ifndef __TUCudaUtility_h
#define __TUCudaUtility_h

#include "TU/CudaArray++.h"
#include <cmath>

namespace TU
{
/************************************************************************
*  3x3 operators							*
************************************************************************/
//! 横方向1階微分オペレータを表す関数オブジェクト
template <class S, class T=S> struct diffH3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	return T(0.5)*(c[2] - c[0]);
    }
};
    
//! 縦方向1階微分オペレータを表す関数オブジェクト
template <class S, class T=S> struct diffV3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	return T(0.5)*(n[1] - p[1]);
    }
};
    
//! 横方向2階微分オペレータを表す関数オブジェクト
template <class S, class T=S> struct diffHH3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	return c[0] - T(2)*c[1] + c[2];
    }
};
    
//! 縦方向2階微分オペレータを表す関数オブジェクト
template <class S, class T=S> struct diffVV3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	return p[1] - T(2)*c[1] + n[1];
    }
};
    
//! 縦横両方向2階微分オペレータを表す関数オブジェクト
template <class S, class T=S> struct diffHV3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	return T(0.25)*(p[0] - p[2] - n[0] + n[2]);
    }
};
    
//! 横方向1階微分Sobelオペレータを表す関数オブジェクト
template <class S, class T=S> struct sobelH3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	return T(0.125)*(p[2] - p[0] + n[2] - n[0]) + T(0.250)*(c[2] - c[0]);
    }
};
    
//! 縦方向1階微分Sobelオペレータを表す関数オブジェクト
template <class S, class T=S> struct sobelV3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	return T(0.125)*(n[0] - p[0] + n[2] - p[2]) + T(0.250)*(n[1] - p[1]);
    }
};
    
//! 1階微分Sobelオペレータの縦横両方向出力の絶対値の和を表す関数オブジェクト
template <class S, class T=S> struct sobelAbs3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	using namespace	std;
	
	return abs(sobelH3x3<S, T>()(p, c, n))
	     + abs(sobelV3x3<S, T>()(p, c, n));
    }
};
    
//! ラプラシアンオペレータを表す関数オブジェクト
template <class S, class T=S> struct laplacian3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	return c[0] + c[2] + p[1] + n[1] - T(4)*c[1];
    }
};
    
//! ヘッセ行列式オペレータを表す関数オブジェクト
template <class S, class T=S> struct det3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	const T	dxy = diffHV3x3<S, T>()(p, c, n);
	
	return diffHH3x3<S, T>()(p, c, n) * diffVV3x3<S, T>()(p, c, n)
	     - dxy * dxy;
    }
};

//! 極大点検出オペレータを表す関数オブジェクト
template <class T> class maximal3x3
{
  public:
    typedef T	value_type;

    __host__ __device__
    maximal3x3(T nonMaximal=0)	:_nonMaximal(nonMaximal)	{}
    
    __host__ __device__ T
    operator ()(const T* p, const T* c, const T* n) const
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
template <class T> class minimal3x3
{
  public:
    typedef T	value_type;

    __host__ __device__
    minimal3x3(T nonMinimal=0)	:_nonMinimal(nonMinimal)	{}
    
    __host__ __device__ T
    operator ()(const T* p, const T* c, const T* n) const
    {
	return ((c[1] < p[0]) && (c[1] < p[1]) && (c[1] < p[2]) &&
		(c[1] < c[0])		       && (c[1] < c[2]) &&
		(c[1] < n[0]) && (c[1] < n[1]) && (c[1] < n[2]) ?
		c[1] : _nonMinimal);
    }

  private:
    const T	_nonMinimal;
};

/************************************************************************
*  utilities								*
************************************************************************/
//! CUDAの定数メモリ領域にデータをコピーする．
/*!
  \param begin	コピー元データの先頭を指す反復子
  \param end	コピー元データの末尾の次を指す反復子
  \param dst	コピー先の定数メモリ領域を指すポインタ
*/
template <class Iterator, class T> inline void
cudaCopyToConstantMemory(Iterator begin, Iterator end, T* dst)
{
    if (begin < end)
	cudaMemcpyToSymbol((const char*)dst, &(*begin),
			   (end - begin)*sizeof(T));
}

template <class T> void
cudaSubsample(const CudaArray2<T>& in, CudaArray2<T>& out)		;

template <class S, class T, class OP> void
cudaOp3x3(const CudaArray2<S>& in, CudaArray2<T>& out, OP op)		;
    
template <class T, class OP> void
cudaSuppressNonExtrema3x3(const CudaArray2<T>& in, CudaArray2<T>& out,
			  OP op, T nulval=0)				;
}

#endif	// !__TUCudaUtility_h
