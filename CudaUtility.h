/*
 *  $Id: CudaUtility.h,v 1.3 2011-04-20 08:15:07 ueshiba Exp $
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
//! ������1����ʬ���ڥ졼����ɽ���ؿ����֥�������
template <class S, class T> struct diffH3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	return T(0.5)*(c[2] - c[0]);
    }
};
    
//! ������1����ʬ���ڥ졼����ɽ���ؿ����֥�������
template <class S, class T> struct diffV3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	return T(0.5)*(n[1] - p[1]);
    }
};
    
//! ������2����ʬ���ڥ졼����ɽ���ؿ����֥�������
template <class S, class T> struct diffHH3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	return c[0] - T(2)*c[1] + c[2];
    }
};
    
//! ������2����ʬ���ڥ졼����ɽ���ؿ����֥�������
template <class S, class T> struct diffVV3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	return p[1] - T(2)*c[1] + n[1];
    }
};
    
//! �Ĳ�ξ����2����ʬ���ڥ졼����ɽ���ؿ����֥�������
template <class S, class T> struct diffHV3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	return T(0.25)*(p[0] - p[2] - n[0] + n[2]);
    }
};
    
//! ������1����ʬSobel���ڥ졼����ɽ���ؿ����֥�������
template <class S, class T> struct sobelH3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	return T(0.125)*(p[2] - p[0] + n[2] - n[0]) + T(0.250)*(c[2] - c[0]);
    }
};
    
//! ������1����ʬSobel���ڥ졼����ɽ���ؿ����֥�������
template <class S, class T> struct sobelV3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	return T(0.125)*(n[0] - p[0] + n[2] - p[2]) + T(0.250)*(n[1] - p[1]);
    }
};
    
//! 1����ʬSobel���ڥ졼���νĲ�ξ�������Ϥ������ͤ��¤�ɽ���ؿ����֥�������
template <class S, class T> struct sobelAbs3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	using namespace	std;
	
	return abs(sobelH3x3<S, T>()(p, c, n))
	     + abs(sobelV3x3<S, T>()(p, c, n));
    }
};
    
//! ��ץ饷���󥪥ڥ졼����ɽ���ؿ����֥�������
template <class S, class T> struct laplacian3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	return c[0] + c[2] + p[1] + n[1] - T(4)*c[1];
    }
};
    
//! �إå����󼰥��ڥ졼����ɽ���ؿ����֥�������
template <class S, class T> struct det3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	const T	dxy = diffHV3x3<S, T>()(p, c, n);
	
	return diffHH3x3<S, T>()(p, c, n) * diffVV3x3<S, T>()(p, c, n)
	     - dxy * dxy;
    }
};

//! ���������Х��ڥ졼����ɽ���ؿ����֥�������
template <class S, class T> struct maximal3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	return T((c[1] > p[0]) & (c[1] > p[1]) & (c[1] > p[2]) &
		 (c[1] > c[0])		       & (c[1] > c[2]) &
		 (c[1] > n[0]) & (c[1] > n[1]) & (c[1] > n[2]));
    }
};

//! �˾������Х��ڥ졼����ɽ���ؿ����֥�������
template <class S, class T> struct minimal3x3
{
    __host__ __device__ T
    operator ()(const S* p, const S* c, const S* n)
    {
	return T((c[1] < p[0]) & (c[1] < p[1]) & (c[1] < p[2]) &
		 (c[1] < c[0])		       & (c[1] < c[2]) &
		 (c[1] < n[0]) & (c[1] < n[1]) & (c[1] < n[2]));
    }
};

/************************************************************************
*  utilities								*
************************************************************************/
//! CUDA����������ΰ�˥ǡ����򥳥ԡ����롥
/*!
  \param begin	���ԡ����ǡ�������Ƭ��ؤ�ȿ����
  \param end	���ԡ����ǡ����������μ���ؤ�ȿ����
  \param dst	���ԡ������������ΰ��ؤ��ݥ���
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
    
}

#endif	// !__TUCudaUtility_h
