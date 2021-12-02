/*
 *  $Id$
 */
/*!
  \file		FIRFilter.h
  \brief	finite impulse responseフィルタの定義と実装
*/
#pragma once

#include "TU/cuda/Array++.h"
#include "TU/cuda/algorithm.h"

namespace TU
{
namespace cuda
{
/************************************************************************
*  class FIRFilter2<T>							*
************************************************************************/
//! CUDAによるseparableな2次元フィルタを表すクラス
template <class T=float, class BLOCK_TRAITS=BlockTraits<> >
class FIRFilter2 : public BLOCK_TRAITS
{
  public:
    using value_type	= T;

    using			BLOCK_TRAITS::BlockDimX;
    using			BLOCK_TRAITS::BlockDimY;
    constexpr static size_t	LobeSizeMax = 17;

  public:
  //! CUDAによる2次元フィルタを生成する．
    FIRFilter2()	:_lobeSizeH(0), _lobeSizeV(0)			{}

    FIRFilter2&	initialize(const TU::Array<float>& lobeH,
			   const TU::Array<float>& lobeV)		;
    template <class IN, class OUT>
    void	convolve(IN ib, IN ie,
			 OUT out, bool shift=false)		const	;

    size_t	outSizeH(size_t inSize)	const	{return inSize - 2*offsetH();}
    size_t	outSizeV(size_t inSize)	const	{return inSize - 2*offsetV();}
    size_t	offsetH()		const	{return _lobeSizeH & ~0x1;}
    size_t	offsetV()		const	{return _lobeSizeV & ~0x1;}

  private:
    template <size_t L, class IN, class OUT>
    static void	convolveH(IN in, IN ie, OUT out)			;
    template <size_t L, class IN, class OUT>
    void	convolveV(IN in, IN ie, OUT out, bool shift)	const	;

  private:
    size_t		_lobeSizeH;	//!< 水平方向フィルタのローブ長
    size_t		_lobeSizeV;	//!< 垂直方向フィルタのローブ長
    mutable Array2<T>	_buf;		//!< 中間結果用のバッファ
};

#if defined(__NVCC__)
namespace device
{
/************************************************************************
*  global __constatnt__ variables					*
************************************************************************/
__constant__ static float	_lobeH[FIRFilter2<>::LobeSizeMax];
__constant__ static float	_lobeV[FIRFilter2<>::LobeSizeMax];

/************************************************************************
*  __device__ functions							*
************************************************************************/
template <class IN, class T> __device__ inline auto
convolve(IN in, const T* lobe, std::integral_constant<size_t, 17>)
{
  // ローブ長が17画素の偶関数畳み込みカーネル
    return lobe[ 0] * (in[ 0] + in[32])
	 + lobe[ 1] * (in[ 1] + in[31])
	 + lobe[ 2] * (in[ 2] + in[30])
	 + lobe[ 3] * (in[ 3] + in[29])
	 + lobe[ 4] * (in[ 4] + in[28])
	 + lobe[ 5] * (in[ 5] + in[27])
	 + lobe[ 6] * (in[ 6] + in[26])
	 + lobe[ 7] * (in[ 7] + in[25])
	 + lobe[ 8] * (in[ 8] + in[24])
	 + lobe[ 9] * (in[ 9] + in[23])
	 + lobe[10] * (in[10] + in[22])
	 + lobe[11] * (in[11] + in[21])
	 + lobe[12] * (in[12] + in[20])
	 + lobe[13] * (in[13] + in[19])
	 + lobe[14] * (in[14] + in[18])
	 + lobe[15] * (in[15] + in[17])
	 + lobe[16] *  in[16];
}

template <class IN, class T> __device__ inline auto
convolve(IN in, const T* lobe, std::integral_constant<size_t, 16>)
{
  // ローブ長が16画素の奇関数畳み込みカーネル
    return lobe[ 0] * (in[ 0] - in[32])
	 + lobe[ 1] * (in[ 1] - in[31])
	 + lobe[ 2] * (in[ 2] - in[30])
	 + lobe[ 3] * (in[ 3] - in[29])
	 + lobe[ 4] * (in[ 4] - in[28])
	 + lobe[ 5] * (in[ 5] - in[27])
	 + lobe[ 6] * (in[ 6] - in[26])
	 + lobe[ 7] * (in[ 7] - in[25])
	 + lobe[ 8] * (in[ 8] - in[24])
	 + lobe[ 9] * (in[ 9] - in[23])
	 + lobe[10] * (in[10] - in[22])
	 + lobe[11] * (in[11] - in[21])
	 + lobe[12] * (in[12] - in[20])
	 + lobe[13] * (in[13] - in[19])
	 + lobe[14] * (in[14] - in[18])
	 + lobe[15] * (in[15] - in[17]);
}

template <class IN, class T> __device__ inline auto
convolve(IN in, const T* lobe, std::integral_constant<size_t, 9>)
{
  // ローブ長が9画素の偶関数畳み込みカーネル
    return lobe[0] * (in[0] + in[16])
	 + lobe[1] * (in[1] + in[15])
	 + lobe[2] * (in[2] + in[14])
	 + lobe[3] * (in[3] + in[13])
	 + lobe[4] * (in[4] + in[12])
	 + lobe[5] * (in[5] + in[11])
	 + lobe[6] * (in[6] + in[10])
	 + lobe[7] * (in[7] + in[ 9])
	 + lobe[8] *  in[8];
}

template <class IN, class T> __device__ inline auto
convolve(IN in, const T* lobe, std::integral_constant<size_t, 8>)
{
  // ローブ長が8画素の奇関数畳み込みカーネル
    return lobe[0] * (in[0] - in[16])
	 + lobe[1] * (in[1] - in[15])
	 + lobe[2] * (in[2] - in[14])
	 + lobe[3] * (in[3] - in[13])
	 + lobe[4] * (in[4] - in[12])
	 + lobe[5] * (in[5] - in[11])
	 + lobe[6] * (in[6] - in[10])
	 + lobe[7] * (in[7] - in[ 9]);
}

template <class IN, class T> __device__ inline auto
convolve(IN in, const T* lobe, std::integral_constant<size_t, 5>)
{
  // ローブ長が5画素の偶関数畳み込みカーネル
    return lobe[0] * (in[0] + in[8])
	 + lobe[1] * (in[1] + in[7])
	 + lobe[2] * (in[2] + in[6])
	 + lobe[3] * (in[3] + in[5])
	 + lobe[4] *  in[4];
}

template <class IN, class T> __device__ inline auto
convolve(IN in, const T* lobe, std::integral_constant<size_t, 4>)
{
  // ローブ長が4画素の奇関数畳み込みカーネル
    return lobe[0] * (in[0] - in[8])
	 + lobe[1] * (in[1] - in[7])
	 + lobe[2] * (in[2] - in[6])
	 + lobe[3] * (in[3] - in[5]);
}

template <class IN, class T> __device__ inline auto
convolve(IN in, const T* lobe, std::integral_constant<size_t, 3>)
{
  // ローブ長が3画素の偶関数畳み込みカーネル
    return lobe[0] * (in[0] + in[4])
	 + lobe[1] * (in[1] + in[3])
	 + lobe[2] *  in[2];
}

template <class IN, class T> __device__ inline auto
convolve(IN in, const T* lobe, std::integral_constant<size_t, 2>)
{
  // ローブ長が2画素の奇関数畳み込みカーネル
    return lobe[0] * (in[0] - in[4])
	 + lobe[1] * (in[1] - in[3]);
}

/************************************************************************
*  __global__ functions							*
************************************************************************/
template <class FILTER, size_t L,
	  class IN, class OUT, class STRIDE_I, class STRIDE_O> __global__ void
fir_filterH(IN in, OUT out, STRIDE_I strideI, STRIDE_O strideO)
{
    using value_type  =	typename FILTER::value_type;

    constexpr auto	LobeSize  = L & ~0x1;	// 中心点を含まないローブ長

    const auto	x0 = __mul24(blockIdx.x, blockDim.x);  // ブロック左上隅
    const auto	y0 = __mul24(blockIdx.y, blockDim.y);  // ブロック左上隅

    advance_stride(in,  y0*strideI);
    advance_stride(out, (y0 + threadIdx.y)*strideO);

  // 原画像のブロックとその左右LobeSize分を共有メモリにコピー
    __shared__ value_type	in_s[FILTER::BlockDimY]
				    [FILTER::BlockDimX + 2*LobeSize + 1];
    loadTileH(in + x0, strideI, in_s, 2*LobeSize);
    __syncthreads();

  // 積和演算
    out[x0 + threadIdx.x] = convolve(&in_s[threadIdx.y][threadIdx.x], _lobeH,
				     std::integral_constant<size_t, L>());
}

template <class FILTER, size_t L,
	  class IN, class OUT, class STRIDE_I, class STRIDE_O> __global__ void
fir_filterV(IN in, OUT out, STRIDE_I strideI, STRIDE_O strideO)
{
    using value_type  =	typename FILTER::value_type;

    constexpr auto	LobeSize  = L & ~0x1;	// 中心点を含まないローブ長

    const auto	x0 = __mul24(blockIdx.x, blockDim.x);  // ブロック左上隅
    const auto	y0 = __mul24(blockIdx.y, blockDim.y);  // ブロック左上隅

    advance_stride(in,  y0*strideI);
    advance_stride(out, (y0 + threadIdx.y)*strideO);

  // 原画像のブロックとその上下LobeSize分を転置して共有メモリにコピー
    __shared__ value_type	in_s[FILTER::BlockDimX]
				    [FILTER::BlockDimY + 2*LobeSize + 1];
    loadTileVt(in + x0, strideI, in_s, 2*LobeSize);
    __syncthreads();

  // 積和演算
    out[x0 + threadIdx.x] = convolve(&in_s[threadIdx.x][threadIdx.y], _lobeV,
				     std::integral_constant<size_t, L>());
}
}	// namespace device

/************************************************************************
*  class FIRFilter2<T, BLOCK_DIM_X, BLOCK_DIM_Y>			*
************************************************************************/
//! 2次元フィルタのローブを設定する．
/*!
  与えるローブの長さは，畳み込みカーネルが偶関数の場合2^n + 1, 奇関数の場合2^n
  (n = 1, 2, 3, 4)でなければならない．
  \param lobeH	横方向ローブ
  \param lobeV	縦方向ローブ
  \return	この2次元フィルタ
*/
template <class T, class BLOCK_TRAITS>
FIRFilter2<T, BLOCK_TRAITS>&
FIRFilter2<T, BLOCK_TRAITS>::initialize(const TU::Array<float>& lobeH,
					const TU::Array<float>& lobeV)
{
    if (lobeH.size() > LobeSizeMax || lobeV.size() > LobeSizeMax)
	throw std::runtime_error("FIRFilter2<T, BLOCK_TRAITS>::initialize: too large lobe size!");

    _lobeSizeH = lobeH.size();
    _lobeSizeV = lobeV.size();
    cudaMemcpyToSymbol(device::_lobeH, lobeH.data(),
		       lobeH.size()*sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(device::_lobeV, lobeV.data(),
		       lobeV.size()*sizeof(float), 0, cudaMemcpyHostToDevice);

    return *this;
}

template <class T, class BLOCK_TRAITS>
template <size_t L, class IN, class OUT> void
FIRFilter2<T, BLOCK_TRAITS>::convolveH(IN in, IN ie, OUT out)
{
    using	std::cbegin;
    using	std::cend;
    using	std::begin;

    constexpr auto	LobeSizeH = L & ~0x1;	// 中心点を含まないローブ長

    const auto	nrow    = std::distance(in, ie);
    const auto	ncol    = std::distance(cbegin(*in), cend(*in)) - 2*LobeSizeH;
    const auto	strideI = stride(in);
    const auto	strideO = stride(out);

  // 左上
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncol/threads.x, nrow/threads.y);
    device::fir_filterH<FIRFilter2, L><<<blocks, threads>>>(cbegin(*in),
							    begin(*out),
							    strideI, strideO);
  // 右上
    const auto	x = blocks.x*threads.x;
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::fir_filterH<FIRFilter2, L><<<blocks, threads>>>(cbegin(*in) + x,
							    begin(*out) + x,
							    strideI, strideO);
  // 左下
    std::advance(in,  blocks.y*threads.y);
    std::advance(out, blocks.y*threads.y);
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    threads.y = nrow%threads.y;
    blocks.y  = 1;
    device::fir_filterH<FIRFilter2, L><<<blocks, threads>>>(cbegin(*in),
							    begin(*out),
							    strideI, strideO);
  // 右下
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::fir_filterH<FIRFilter2, L><<<blocks, threads>>>(cbegin(*in) + x,
							    begin(*out) + x,
							    strideI, strideO);
}

template <class T, class BLOCK_TRAITS>
template <size_t L, class IN, class OUT> void
FIRFilter2<T, BLOCK_TRAITS>::convolveV(IN in, IN ie, OUT out, bool shift) const
{
    using	std::cbegin;
    using	std::cend;
    using	std::begin;

    const auto	nrow	  = outSizeV(std::distance(in, ie));
    const auto	ncol	  = std::distance(cbegin(*in), cend(*in));
    const auto	strideI   = stride(in);
    const auto	strideO   = stride(out);
    size_t	dx	  = 0;

    if (shift)
    {
	out += offsetV();
	dx   = offsetH();
    }

  // 左上
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncol/threads.x, nrow/threads.y);
    device::fir_filterV<FIRFilter2, L><<<blocks, threads>>>(cbegin(*in),
							    begin(*out) + dx,
							    strideI, strideO);
  // 右上
    const auto	x = blocks.x*threads.x;
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::fir_filterV<FIRFilter2, L><<<blocks, threads>>>(cbegin(*in) + x,
							    begin(*out) + x + dx,
							    strideI, strideO);
  // 左下
    std::advance(in,  blocks.y*threads.y);
    std::advance(out, blocks.y*threads.y);
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    threads.y = nrow%threads.y;
    blocks.y  = 1;
    device::fir_filterV<FIRFilter2, L><<<blocks, threads>>>(cbegin(*in),
							    begin(*out) + dx,
							    strideI, strideO);
  // 右下
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::fir_filterV<FIRFilter2, L><<<blocks, threads>>>(cbegin(*in) + x,
							    begin(*out) + x + dx,
							    strideI, strideO);
}
#endif	// __NVCC__

//! 与えられた2次元配列とこのフィルタを畳み込む
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class T, class BLOCK_TRAITS> template <class IN, class OUT> void
FIRFilter2<T, BLOCK_TRAITS>::convolve(IN in, IN ie, OUT out, bool shift) const
{
    using	std::cbegin;
    using	std::cend;

    const size_t	nrow = std::distance(in, ie);
    if (nrow < 4*(_lobeSizeV/2) + 1)
	return;

    const size_t	ncol = std::distance(cbegin(*in), cend(*in));
    if (ncol < 4*(_lobeSizeH/2) + 1)
	return;

    _buf.resize(nrow, ncol - 4*(_lobeSizeH/2) + 1);

  // 横方向に畳み込む．
    switch (_lobeSizeH)
    {
      case 17:
	convolveH<17>(in, ie, _buf.begin());
	break;
      case 16:
	convolveH<16>(in, ie, _buf.begin());
	break;
      case  9:
	convolveH< 9>(in, ie, _buf.begin());
	break;
      case  8:
	convolveH< 8>(in, ie, _buf.begin());
	break;
      case  5:
	convolveH< 5>(in, ie, _buf.begin());
	break;
      case  4:
	convolveH< 4>(in, ie, _buf.begin());
	break;
      case  3:
	convolveH< 3>(in, ie, _buf.begin());
	break;
      case  2:
	convolveH< 2>(in, ie, _buf.begin());
	break;
      default:
	throw std::runtime_error("FIRFilter2::convolve: unsupported horizontal lobe size!");
    }

  // 縦方向に畳み込む．
    switch (_lobeSizeV)
    {
      case 17:
	convolveV<17>(_buf.begin(), _buf.end(), out, shift);
	break;
      case 16:
	convolveV<16>(_buf.begin(), _buf.end(), out, shift);
	break;
      case  9:
	convolveV< 9>(_buf.begin(), _buf.end(), out, shift);
	break;
      case  8:
	convolveV< 8>(_buf.begin(), _buf.end(), out, shift);
	break;
      case  5:
	convolveV< 5>(_buf.begin(), _buf.end(), out, shift);
	break;
      case  4:
	convolveV< 4>(_buf.begin(), _buf.end(), out, shift);
	break;
      case  3:
	convolveV< 3>(_buf.begin(), _buf.end(), out, shift);
	break;
      case  2:
	convolveV< 2>(_buf.begin(), _buf.end(), out, shift);
	break;
      default:
	throw std::runtime_error("FIRFilter2::convolve: unsupported vertical lobe size!");
    }
}

}
}
