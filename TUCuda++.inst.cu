/*
 *  $Id$
 */
/*!
  \file		TUCuda++.inst.cu
  \brief	繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝�縺ｮ螳滉ｽ灘喧
*/
#include "TU/cuda/Array++.h"
#include "TU/cuda/algorithm.h"

namespace TU
{
namespace cuda
{
template void
subsample(Array2<u_char>::const_iterator in,
	  Array2<u_char>::const_iterator ie,
	  Array2<u_char>::iterator out)					;
template void
subsample(Array2<float>::const_iterator in,
	  Array2<float>::const_iterator ie,
	  Array2<float>::iterator out)					;
  /*
template void
suppressNonExtrema3x3(Array2<u_char>::const_iterator in,
		      Array2<u_char>::const_iterator ie,
		      Array2<u_char>::iterator out,
		      thrust::greater<u_char> op, u_char nulval)	;
template void
suppressNonExtrema3x3(Array2<float>::const_iterator in,
		      Array2<float>::const_iterator ie,
		      Array2<float>::iterator out,
		      thrust::greater<float> op, float nulval)		;
template void
suppressNonExtrema3x3(Array2<u_char>::const_iterator in,
		      Array2<u_char>::const_iterator ie,
		      Array2<u_char>::iterator out,
		      thrust::less<u_char> op, u_char nulval)		;
template void
suppressNonExtrema3x3(Array2<float>::const_iterator in,
		      Array2<float>::const_iterator ie,
		      Array2<float>::iterator out,
		      thrust::less<float> op, float nulval)		;
  */
}	// namespace cuda
}	// namespace TU

