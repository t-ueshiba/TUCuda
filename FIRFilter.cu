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
namespace device
{
/************************************************************************
*  global __constatnt__ variables					*
************************************************************************/
static __constant__ float	_lobeH[FIRFilter2::LobeSizeMax];
static __constant__ float	_lobeV[FIRFilter2::LobeSizeMax];

/************************************************************************
*  global __device__ functions						*
************************************************************************/
__host__ __device__ const float*	lobeH()		{ return _lobeH; }
__host__ __device__ const float*	lobeV()		{ return _lobeV; }

}	// namespace device
    
//! 2次元フィルタのローブを設定する．
/*!
  与えるローブの長さは，畳み込みカーネルが偶関数の場合2^n + 1, 奇関数の場合2^n
  (n = 1, 2, 3, 4)でなければならない．
  \param lobeH	横方向ローブ
  \param lobeV	縦方向ローブ
  \return	この2次元フィルタ
*/
FIRFilter2&
FIRFilter2::initialize(const TU::Array<float>& lobeH,
		       const TU::Array<float>& lobeV)
{
    if (lobeH.size() > LobeSizeMax || lobeV.size() > LobeSizeMax)
	throw std::runtime_error("FIRFilter2::initialize: too large lobe size!");
    
    _lobeSizeH = lobeH.size();
    _lobeSizeV = lobeV.size();
    cudaMemcpyToSymbol(device::_lobeH, lobeH.data(),
		       lobeH.size()*sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(device::_lobeV, lobeV.data(),
		       lobeV.size()*sizeof(float), 0, cudaMemcpyHostToDevice);

    return *this;
}

/************************************************************************
*  instantiations							*
************************************************************************/
template void
FIRFilter2::convolve(Array2<u_char>::const_iterator in,
		     Array2<u_char>::const_iterator ie,
		     Array2<u_char>::iterator out)		const	;
template void
FIRFilter2::convolve(Array2<u_char>::const_iterator in,
		     Array2<u_char>::const_iterator ie,
		     Array2<float>::iterator out)		const	;
template void
FIRFilter2::convolve(Array2<float>::const_iterator in,
		     Array2<float>::const_iterator ie,
		     Array2<u_char>::iterator out)		const	;
template void
FIRFilter2::convolve(Array2<float>::const_iterator in,
		     Array2<float>::const_iterator ie,
		     Array2<float>::iterator out)		const	;
}
}
