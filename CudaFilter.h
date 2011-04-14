/*
 *  $Id: CudaFilter.h,v 1.2 2011-04-14 08:39:34 ueshiba Exp $
 */
#ifndef __TUCudaFilter_h
#define __TUCudaFilter_h

#include "TU/CudaArray++.h"

namespace TU
{
/************************************************************************
*  class CudaFilter2							*
************************************************************************/
//! CUDA�ˤ��separable��2�����ե��륿��ɽ�����饹
class CudaFilter2
{
  public:
    enum		{LOBE_SIZE_MAX = 17};

  public:
    CudaFilter2()							;
    
    CudaFilter2&	initialize(const Array<float>& lobeH,
				   const Array<float>& lobeV)		;
    template <class S, class T>
    const CudaFilter2&	convolve(const CudaArray2<S>& in,
				       CudaArray2<T>& out)	const	;
    template <class S, class T>
    const CudaFilter2&	convolveH(const CudaArray2<S>& in,
					CudaArray2<T>& out)	const	;
    template <class S, class T>
    const CudaFilter2&	convolveV(const CudaArray2<S>& in,
					CudaArray2<T>& out)	const	;

  private:
    cudaDeviceProp		_prop;		//!< �ǥХ���������
    u_int			_lobeSizeH;	//!< ��ʿ�����ե��륿�Υ���Ĺ
    u_int			_lobeSizeV;	//!< ��ľ�����ե��륿�Υ���Ĺ
    mutable CudaArray2<float>	_buf;		//!< ��ַ���ѤΥХåե�
};
    
}

#endif	// !__TUCudaFilter_h
