/*
 *  $Id$
 */
/*!
  \file		FIRFilter.h
  \brief	finite impulse responseフィルタの定義と実装
*/ 
#ifndef TU_CUDA_FIRFILTER_H
#define TU_CUDA_FIRFILTER_H

#include "TU/cuda/Array++.h"
#include "TU/cuda/algorithm.h"

namespace TU
{
namespace cuda
{
/************************************************************************
*  class FIRFilter2							*
************************************************************************/
//! CUDAによるseparableな2次元フィルタを表すクラス
class FIRFilter2
{
  public:
  //! CUDAによる2次元フィルタを生成する．
    FIRFilter2()	:_lobeSizeH(0), _lobeSizeV(0)			{}
    
    FIRFilter2&	initialize(const TU::Array<float>& lobeH,
			   const TU::Array<float>& lobeV)		;
    template <class IN, class OUT>
    void	convolve(IN ib, IN ie, OUT out)			const	;

  private:
    template <size_t L, class IN, class OUT>
    static void	convolveH(IN in, IN ie, OUT out)			;
    template <size_t L, class IN, class OUT>
    void	convolveV(IN in, IN ie, OUT out)		const	;
    
  public:
    static constexpr size_t	LobeSizeMax = 17;
    static constexpr size_t	BlockDimX   = 32;
    static constexpr size_t	BlockDimY   = 16;
    static constexpr size_t	BlockDim    = 16;

  private:
    size_t			_lobeSizeH;	//!< 水平方向フィルタのローブ長
    size_t			_lobeSizeV;	//!< 垂直方向フィルタのローブ長
    mutable Array2<float>	_buf;		//!< 中間結果用のバッファ
};

#if defined(__NVCC__)
namespace device
{
/************************************************************************
*  global __constatnt__ variables					*
************************************************************************/
__constant__ static float	_lobeH[FIRFilter2::LobeSizeMax];
__constant__ static float	_lobeV[FIRFilter2::LobeSizeMax];

/************************************************************************
*  __device__ functions							*
************************************************************************/
template <size_t L> __device__ static float
convolve(const float* in_s, const float* lobe)	;
    
template <> __device__ inline float
convolve<17>(const float* in_s, const float* lobe)
{
  // ローブ長が17画素の偶関数畳み込みカーネル
    return lobe[ 0] * (in_s[ 0] + in_s[32])
	 + lobe[ 1] * (in_s[ 1] + in_s[31])
	 + lobe[ 2] * (in_s[ 2] + in_s[30])
	 + lobe[ 3] * (in_s[ 3] + in_s[29])
	 + lobe[ 4] * (in_s[ 4] + in_s[28])
	 + lobe[ 5] * (in_s[ 5] + in_s[27])
	 + lobe[ 6] * (in_s[ 6] + in_s[26])
	 + lobe[ 7] * (in_s[ 7] + in_s[25])
	 + lobe[ 8] * (in_s[ 8] + in_s[24])
	 + lobe[ 9] * (in_s[ 9] + in_s[23])
	 + lobe[10] * (in_s[10] + in_s[22])
	 + lobe[11] * (in_s[11] + in_s[21])
	 + lobe[12] * (in_s[12] + in_s[20])
	 + lobe[13] * (in_s[13] + in_s[19])
	 + lobe[14] * (in_s[14] + in_s[18])
	 + lobe[15] * (in_s[15] + in_s[17])
	 + lobe[16] *  in_s[16];
}
    
template <> __device__ inline float
convolve<16>(const float* in_s, const float* lobe)
{
  // ローブ長が16画素の奇関数畳み込みカーネル
    return lobe[ 0] * (in_s[ 0] - in_s[32])
	 + lobe[ 1] * (in_s[ 1] - in_s[31])
	 + lobe[ 2] * (in_s[ 2] - in_s[30])
	 + lobe[ 3] * (in_s[ 3] - in_s[29])
	 + lobe[ 4] * (in_s[ 4] - in_s[28])
	 + lobe[ 5] * (in_s[ 5] - in_s[27])
	 + lobe[ 6] * (in_s[ 6] - in_s[26])
	 + lobe[ 7] * (in_s[ 7] - in_s[25])
	 + lobe[ 8] * (in_s[ 8] - in_s[24])
	 + lobe[ 9] * (in_s[ 9] - in_s[23])
	 + lobe[10] * (in_s[10] - in_s[22])
	 + lobe[11] * (in_s[11] - in_s[21])
	 + lobe[12] * (in_s[12] - in_s[20])
	 + lobe[13] * (in_s[13] - in_s[19])
	 + lobe[14] * (in_s[14] - in_s[18])
	 + lobe[15] * (in_s[15] - in_s[17]);
}
    
template <> __device__ inline float
convolve<9>(const float* in_s, const float* lobe)
{
  // ローブ長が9画素の偶関数畳み込みカーネル
    return lobe[0] * (in_s[0] + in_s[16])
	 + lobe[1] * (in_s[1] + in_s[15])
	 + lobe[2] * (in_s[2] + in_s[14])
	 + lobe[3] * (in_s[3] + in_s[13])
	 + lobe[4] * (in_s[4] + in_s[12])
	 + lobe[5] * (in_s[5] + in_s[11])
	 + lobe[6] * (in_s[6] + in_s[10])
	 + lobe[7] * (in_s[7] + in_s[ 9])
	 + lobe[8] *  in_s[8];
}
    
template <> __device__ inline float
convolve<8>(const float* in_s, const float* lobe)
{
  // ローブ長が8画素の奇関数畳み込みカーネル
    return lobe[0] * (in_s[0] - in_s[16])
	 + lobe[1] * (in_s[1] - in_s[15])
	 + lobe[2] * (in_s[2] - in_s[14])
	 + lobe[3] * (in_s[3] - in_s[13])
	 + lobe[4] * (in_s[4] - in_s[12])
	 + lobe[5] * (in_s[5] - in_s[11])
	 + lobe[6] * (in_s[6] - in_s[10])
	 + lobe[7] * (in_s[7] - in_s[ 9]);
}
    
template <> __device__ inline float
convolve<5>(const float* in_s, const float* lobe)
{
  // ローブ長が5画素の偶関数畳み込みカーネル
    return lobe[0] * (in_s[0] + in_s[8])
	 + lobe[1] * (in_s[1] + in_s[7])
	 + lobe[2] * (in_s[2] + in_s[6])
	 + lobe[3] * (in_s[3] + in_s[5])
	 + lobe[4] *  in_s[4];
}
    
template <> __device__ inline float
convolve<4>(const float* in_s, const float* lobe)
{
  // ローブ長が4画素の奇関数畳み込みカーネル
    return lobe[0] * (in_s[0] - in_s[8])
	 + lobe[1] * (in_s[1] - in_s[7])
	 + lobe[2] * (in_s[2] - in_s[6])
	 + lobe[3] * (in_s[3] - in_s[5]);
}
    
template <> __device__ inline float
convolve<3>(const float* in_s, const float* lobe)
{
  // ローブ長が3画素の偶関数畳み込みカーネル
    return lobe[0] * (in_s[0] + in_s[4])
	 + lobe[1] * (in_s[1] + in_s[3])
	 + lobe[2] *  in_s[2];
}
    
template <> __device__ inline float
convolve<2>(const float* in_s, const float* lobe)
{
  // ローブ長が2画素の奇関数畳み込みカーネル
    return lobe[0] * (in_s[0] - in_s[4])
	 + lobe[1] * (in_s[1] - in_s[3]);    
}
    
/************************************************************************
*  __global__ functions							*
************************************************************************/
template <size_t L, class IN, class OUT> __global__ static void
fir_filterH(IN in, OUT out, int stride_i, int stride_o)
{
    constexpr auto	LobeSize = L & ~0x1;	// 中心点を含まないローブ長
    constexpr auto	BlockDimX = FIRFilter2::BlockDimX;
    constexpr auto	BlockDimY = FIRFilter2::BlockDimY;

    const auto	x0 = __mul24(blockIdx.x, blockDim.x);  // ブロック左上隅
    const auto	y0 = __mul24(blockIdx.y, blockDim.y);  // ブロック左上隅

  // 原画像のブロックとその左右LobeSize分を共有メモリにコピー
    __shared__ float	in_s[BlockDimY][BlockDimX + 2*LobeSize + 1];
    loadTileH(in + __mul24(y0, stride_i) + x0, stride_i, in_s, 2*LobeSize);
    __syncthreads();
    
  // 積和演算
    out[__mul24(y0 + threadIdx.y, stride_o) + x0 + threadIdx.x]
	= convolve<L>(&in_s[threadIdx.y][threadIdx.x], _lobeH);
}

template <size_t L, class IN, class OUT> __global__ static void
fir_filterV(const IN in, OUT out, int stride_i, int stride_o)
{
    constexpr auto	LobeSize = L & ~0x1;	// 中心点を含まないローブ長
    constexpr auto	BlockDimX = FIRFilter2::BlockDimX;
    constexpr auto	BlockDimY = FIRFilter2::BlockDimY;

    const auto	x0 = __mul24(blockIdx.x, blockDim.x);  // ブロック左上隅
    const auto	y0 = __mul24(blockIdx.y, blockDim.y);  // ブロック左上隅

  // 原画像のブロックとその左右LobeSize分を共有メモリにコピー
    __shared__ float	in_s[BlockDimX][BlockDimY + 2*LobeSize + 1];
    loadTileVt(in + __mul24(y0, stride_i) + x0, stride_i, in_s, 2*LobeSize);
    __syncthreads();
    
  // 積和演算
    out[__mul24(y0 + threadIdx.y, stride_o) + x0 + threadIdx.x]
	= convolve<L>(&in_s[threadIdx.x][threadIdx.y], _lobeV);
}
}	// namespace device
    
/************************************************************************
*  class FIRFilter2							*
************************************************************************/
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

template <size_t LH, class IN, class OUT> void
FIRFilter2::convolveH(IN in, IN ie, OUT out)
{
    constexpr size_t	LobeSizeH = LH & ~0x1;	// 中心点を含まないローブ長

    const auto	nrow = std::distance(in, ie);
    const auto	ncol = std::distance(std::cbegin(*in), std::cend(*in))
		     - 2*LobeSizeH;
    const auto	stride_i = stride(in);
    const auto	stride_o = stride(out);

  // 左上
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncol/threads.x, nrow/threads.y);
    device::fir_filterH<LH><<<blocks, threads>>>(
	std::cbegin(*in).get(), std::begin(*out).get(), stride_i, stride_o);
  // 右上
    const auto	x = blocks.x*threads.x;
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::fir_filterH<LH><<<blocks, threads>>>(
	std::cbegin(*in).get() + x, std::begin(*out).get() + x,
	stride_i, stride_o);
  // 左下
    std::advance(in,  blocks.y*threads.y);
    std::advance(out, blocks.y*threads.y);
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    threads.y = nrow%threads.y;
    blocks.y  = 1;
    device::fir_filterH<LH><<<blocks, threads>>>(
	std::cbegin(*in).get(), std::begin(*out).get(), stride_i, stride_o);
  // 右下
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::fir_filterH<LH><<<blocks, threads>>>(
	std::cbegin(*in).get() + x, std::begin(*out).get() + x,
	stride_i, stride_o);
}

template <size_t LV, class IN, class OUT> void
FIRFilter2::convolveV(IN in, IN ie, OUT out) const
{
    constexpr auto	LobeSizeV = LV & ~0x1;	// 中心点を含まないローブ長
    
    const auto	nrow	  = std::distance(in, ie) - 2*LobeSizeV;
    const auto	ncol	  = std::distance(std::cbegin(*in), std::cend(*in));
    const auto	stride_i  = stride(in);
    const auto	stride_o  = stride(out);
    const auto	lobeSizeH = _lobeSizeH & ~0x1;	// 中心点を含まないローブ長

  // 左上
    std::advance(out, LobeSizeV);
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncol/threads.x, nrow/threads.y);
    device::fir_filterV<LV><<<blocks, threads>>>(
	std::cbegin(*in).get(), std::begin(*out).get() + lobeSizeH,
	stride_i, stride_o);
  // 右上
    const auto	x = blocks.x*threads.x;
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::fir_filterV<LV><<<blocks, threads>>>(
	std::cbegin(*in).get() + x, std::begin(*out).get() + x + lobeSizeH,
	stride_i, stride_o);
  // 左下
    std::advance(in,  blocks.y*threads.y);
    std::advance(out, blocks.y*threads.y);
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    threads.y = nrow%threads.y;
    blocks.y  = 1;
    device::fir_filterV<LV><<<blocks, threads>>>(
	std::cbegin(*in).get(), std::begin(*out).get() + lobeSizeH,
	stride_i, stride_o);
  // 右下
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::fir_filterV<LV><<<blocks, threads>>>(
	std::cbegin(*in).get() + x, std::begin(*out).get() + x + lobeSizeH,
	stride_i, stride_o);
}
#endif	// __NVCC__

//! 与えられた2次元配列とこのフィルタを畳み込む
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class IN, class OUT> void
FIRFilter2::convolve(IN in, IN ie, OUT out) const
{
    const auto	nrow = std::distance(in, ie);
    if (nrow < 4*(_lobeSizeV/2) + 1)
	return;

    const auto	ncol = std::distance(std::cbegin(*in), std::cend(*in));
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
	convolveV<17>(_buf.begin(), _buf.end(), out);
	break;
      case 16:
	convolveV<16>(_buf.begin(), _buf.end(), out);
	break;
      case  9:
	convolveV< 9>(_buf.begin(), _buf.end(), out);
	break;
      case  8:
	convolveV< 8>(_buf.begin(), _buf.end(), out);
	break;
      case  5:
	convolveV< 5>(_buf.begin(), _buf.end(), out);
	break;
      case  4:
	convolveV< 4>(_buf.begin(), _buf.end(), out);
	break;
      case  3:
	convolveV< 3>(_buf.begin(), _buf.end(), out);
	break;
      case  2:
	convolveV< 2>(_buf.begin(), _buf.end(), out);
	break;
      default:
	throw std::runtime_error("FIRFilter2::convolve: unsupported vertical lobe size!");
    }
}

}
}
#endif	// !TU_CUDA_FIRFILTER_H
