/*
 *  $Id: CudaUtility.h,v 1.1 2011-04-18 08:16:55 ueshiba Exp $
 */
#ifndef __TUCudaUtility_h
#define __TUCudaUtility_h

#include "TU/CudaArray++.h"

namespace TU
{
/************************************************************************
*  3x3 operators							*
************************************************************************/
template <class T> struct diffH3x3
{
    __host__ __device__ T
    operator ()(const T* p, const T* c, const T* n)
    {
	return T(0.5)*(c[2] - c[0]);
    }
};
    
template <class T> struct diffV3x3
{
    __host__ __device__ T
    operator ()(const T* p, const T* c, const T* n)
    {
	return T(0.5)*(n[1] - p[1]);
    }
};
    
template <class T> struct diffHH3x3
{
    __host__ __device__ T
    operator ()(const T* p, const T* c, const T* n)
    {
	return c[0] - T(2)*c[1] + c[2];
    }
};
    
template <class T> struct diffVV3x3
{
    __host__ __device__ T
    operator ()(const T* p, const T* c, const T* n)
    {
	return p[1] - T(2)*c[1] + n[1];
    }
};
    
template <class T> struct diffHV3x3
{
    __host__ __device__ T
    operator ()(const T* p, const T* c, const T* n)
    {
	return T(0.25)*(p[0] - p[2] - n[0] + n[2]);
    }
};
    
template <class T> struct sobelH3x3
{
    __host__ __device__ T
    operator ()(const T* p, const T* c, const T* n)
    {
	return T(0.125)*(p[2] - p[0] + n[2] - n[0]) + T(0.250)*(c[2] - c[0]);
    }
};
    
template <class T> struct sobelV3x3
{
    __host__ __device__ T
    operator ()(const T* p, const T* c, const T* n)
    {
	return T(0.125)*(n[0] - p[0] + n[2] - p[2]) + T(0.250)*(n[1] - p[1]);
    }
};
    
template <class T> struct laplacian3x3
{
    __host__ __device__ T
    operator ()(const T* p, const T* c, const T* n)
    {
	return c[0] + c[2] + p[1] + n[1] - T(4)*c[1];
    }
};
    
template <class T> struct det3x3
{
    __host__ __device__ T
    operator ()(const T* p, const T* c, const T* n)
    {
	const T	dxy = diffHV3x3<float>()(p, c, n);
	
	return diffHH3x3<float>()(p, c, n) * diffVV3x3<float>()(p, c, n)
	     - dxy * dxy;
    }
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

template <class T, class OP> void
cudaOp3x3(const CudaArray2<T>& in, CudaArray2<float>& out, OP op)	;
    
}

#endif	// !__TUCudaUtility_h
