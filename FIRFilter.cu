/*
 * $Id: FIRFilter.cu,v 1.7 2011-04-26 06:39:19 ueshiba Exp $
 */
/*!
  \file		FIRFilter.cu
  \brief	finite impulse responseフィルタの実装
*/
#include "TU/cuda/FIRFilter.h"

namespace TU
{
namespace cuda
{
/************************************************************************
*  instantiations							*
************************************************************************/
template void
FIRFilter2<>::convolve(Array2<u_char>::const_iterator in,
		       Array2<u_char>::const_iterator ie,
		       Array2<u_char>::iterator out,
		       bool shift)				const	;
template void
FIRFilter2<>::convolve(Array2<u_char>::const_iterator in,
		       Array2<u_char>::const_iterator ie,
		       Array2<float>::iterator out,
		       bool shift)				const	;
template void
FIRFilter2<>::convolve(Array2<float>::const_iterator in,
		       Array2<float>::const_iterator ie,
		       Array2<u_char>::iterator out,
		       bool shift)				const	;
template void
FIRFilter2<>::convolve(Array2<float>::const_iterator in,
		       Array2<float>::const_iterator ie,
		       Array2<float>::iterator out,
		       bool shift)				const	;
}
}
