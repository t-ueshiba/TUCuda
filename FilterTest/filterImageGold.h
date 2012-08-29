/*
 *  $Id: filterImageGold.h,v 1.2 2012-08-29 22:43:57 ueshiba Exp $
 */
#include "TU/Image++.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
template <class T1, class B1, class R1, class T2, class B2, class R2> void
filter1D(const Array2<T1, B1, R1>& in, Array2<T2, B2, R2>& out,
	 const Array<float>& coeff)
{
    out.resize(in.ncol(), in.nrow());
    out = 0;

    const u_int	tailWidth = coeff.dim() - 1;
    
    for (u_int i = 0; i < in.nrow(); ++i)
    {
	const T1&	row = in[i];
	
	for (u_int j = tailWidth; j < in.ncol() - tailWidth; ++j)
	{
	  // 積和演算を行う．
	    float	val = coeff[tailWidth] * row[j];
	    for (u_int n = 0; n < tailWidth; ++n)
		val += coeff[n] * (row[j - tailWidth + n] +
				   row[j + tailWidth - n]);

	    out[j][i] = val;
	}
    }
}

/************************************************************************
*  global functions							*
************************************************************************/
template <class S, class T> void
filterImageGold(const Image<S>& original, Image<T>& filtered,
		const Array<float>& coeff)
{
    Array2<Array<float> >	mid;
    filter1D(original, mid, coeff);
    filter1D(mid, filtered, coeff);
}
    
}
