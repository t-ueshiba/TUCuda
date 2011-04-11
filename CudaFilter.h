/*
 *  $Id: CudaFilter.h,v 1.1 2011-04-11 08:10:06 ueshiba Exp $
 */
#ifndef __TUCudaFilter_h
#define __TUCudaFilter_h

#include "TU/CudaArray++.h"

namespace TU
{
/************************************************************************
*  class CudaFilter2							*
************************************************************************/
//! CUDAによる有限長のローブを持つ2次元フィルタを表すクラス
class CudaFilter2
{
  public:
    enum		{LOBE_SIZE_MAX = 17};

  public:
    CudaFilter2&	initialize(const Array<float>& coeffH,
				   const Array<float>& coeffV)		;
    template <class S, class T>
    const CudaFilter2&	convolve(const CudaArray2<S>& in,
				       CudaArray2<T>& out)	const	;

  private:
    template <u_int D, class S, class T>
    void		convolveHorV(const CudaArray2<S>& in,
					   CudaArray2<T>& out)	const	;
    
  private:
    u_int			_lobeSizeH;
    u_int			_lobeSizeV;
    mutable CudaArray2<float>	_buf;
};
    
}

#endif	// !__TUCudaFilter_h
