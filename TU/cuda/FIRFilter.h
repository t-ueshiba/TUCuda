/*
 *  $Id$
 */
/*!
  \file		FIRFilter.h
  \brief	finite impulse responseフィルタの定義と実装
*/ 
#ifndef __TU_CUDA_FIRFILTER_H
#define __TU_CUDA_FIRFILTER_H

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
    static void	convolveV(IN in, IN ie, OUT out)			;
    
  public:
    static constexpr size_t	LobeSizeMax = 17;

  private:
    size_t			_lobeSizeH;	//!< 水平方向フィルタのローブ長
    size_t			_lobeSizeV;	//!< 垂直方向フィルタのローブ長
    mutable Array2<float>	_buf;		//!< 中間結果用のバッファ
};

#if defined(__NVCC__)
/************************************************************************
*  __device__ functions							*
************************************************************************/
__host__ __device__ const float*	lobeH()				;
__host__ __device__ const float*	lobeV()				;

template <size_t L> static __device__ float
convolve(const float* in_s, const float* lobe)	;
    
template <> inline __device__ float
convolve<17>(const float* in_s, const float* lobe)
{
  // ローブ長が17画素の偶関数畳み込みカーネル
    return lobe[ 0] * (in_s[-16] + in_s[16])
	 + lobe[ 1] * (in_s[-15] + in_s[15])
	 + lobe[ 2] * (in_s[-14] + in_s[14])
	 + lobe[ 3] * (in_s[-13] + in_s[13])
	 + lobe[ 4] * (in_s[-12] + in_s[12])
	 + lobe[ 5] * (in_s[-11] + in_s[11])
	 + lobe[ 6] * (in_s[-10] + in_s[10])
	 + lobe[ 7] * (in_s[ -9] + in_s[ 9])
	 + lobe[ 8] * (in_s[ -8] + in_s[ 8])
	 + lobe[ 9] * (in_s[ -7] + in_s[ 7])
	 + lobe[10] * (in_s[ -6] + in_s[ 6])
	 + lobe[11] * (in_s[ -5] + in_s[ 5])
	 + lobe[12] * (in_s[ -4] + in_s[ 4])
	 + lobe[13] * (in_s[ -3] + in_s[ 3])
	 + lobe[14] * (in_s[ -2] + in_s[ 2])
	 + lobe[15] * (in_s[ -1] + in_s[ 1])
	 + lobe[16] *  in_s[  0];
}
    
template <> inline __device__ float
convolve<16>(const float* in_s, const float* lobe)
{
  // ローブ長が16画素の奇関数畳み込みカーネル
    return lobe[ 0] * (in_s[-16] - in_s[16])
	 + lobe[ 1] * (in_s[-15] - in_s[15])
	 + lobe[ 2] * (in_s[-14] - in_s[14])
	 + lobe[ 3] * (in_s[-13] - in_s[13])
	 + lobe[ 4] * (in_s[-12] - in_s[12])
	 + lobe[ 5] * (in_s[-11] - in_s[11])
	 + lobe[ 6] * (in_s[-10] - in_s[10])
	 + lobe[ 7] * (in_s[ -9] - in_s[ 9])
	 + lobe[ 8] * (in_s[ -8] - in_s[ 8])
	 + lobe[ 9] * (in_s[ -7] - in_s[ 7])
	 + lobe[10] * (in_s[ -6] - in_s[ 6])
	 + lobe[11] * (in_s[ -5] - in_s[ 5])
	 + lobe[12] * (in_s[ -4] - in_s[ 4])
	 + lobe[13] * (in_s[ -3] - in_s[ 3])
	 + lobe[14] * (in_s[ -2] - in_s[ 2])
	 + lobe[15] * (in_s[ -1] - in_s[ 1]);
}
    
template <> inline __device__ float
convolve<9>(const float* in_s, const float* lobe)
{
  // ローブ長が9画素の偶関数畳み込みカーネル
    return lobe[0] * (in_s[-8] + in_s[8])
	 + lobe[1] * (in_s[-7] + in_s[7])
	 + lobe[2] * (in_s[-6] + in_s[6])
	 + lobe[3] * (in_s[-5] + in_s[5])
	 + lobe[4] * (in_s[-4] + in_s[4])
	 + lobe[5] * (in_s[-3] + in_s[3])
	 + lobe[6] * (in_s[-2] + in_s[2])
	 + lobe[7] * (in_s[-1] + in_s[1])
	 + lobe[8] *  in_s[ 0];
}
    
template <> inline __device__ float
convolve<8>(const float* in_s, const float* lobe)
{
  // ローブ長が8画素の奇関数畳み込みカーネル
    return lobe[0] * (in_s[-8] - in_s[8])
	 + lobe[1] * (in_s[-7] - in_s[7])
	 + lobe[2] * (in_s[-6] - in_s[6])
	 + lobe[3] * (in_s[-5] - in_s[5])
	 + lobe[4] * (in_s[-4] - in_s[4])
	 + lobe[5] * (in_s[-3] - in_s[3])
	 + lobe[6] * (in_s[-2] - in_s[2])
	 + lobe[7] * (in_s[-1] - in_s[1]);
}
    
template <> inline __device__ float
convolve<5>(const float* in_s, const float* lobe)
{
  // ローブ長が5画素の偶関数畳み込みカーネル
    return lobe[0] * (in_s[-4] + in_s[4])
	 + lobe[1] * (in_s[-3] + in_s[3])
	 + lobe[2] * (in_s[-2] + in_s[2])
	 + lobe[3] * (in_s[-1] + in_s[1])
	 + lobe[4] *  in_s[ 0];
}
    
template <> inline __device__ float
convolve<4>(const float* in_s, const float* lobe)
{
  // ローブ長が4画素の奇関数畳み込みカーネル
    return lobe[0] * (in_s[-4] - in_s[4])
	 + lobe[1] * (in_s[-3] - in_s[3])
	 + lobe[2] * (in_s[-2] - in_s[2])
	 + lobe[3] * (in_s[-1] - in_s[1]);
}
    
template <> inline __device__ float
convolve<3>(const float* in_s, const float* lobe)
{
  // ローブ長が3画素の偶関数畳み込みカーネル
    return lobe[0] * (in_s[-2] + in_s[2])
	 + lobe[1] * (in_s[-1] + in_s[1])
	 + lobe[2] *  in_s[ 0];
}
    
template <> inline __device__ float
convolve<2>(const float* in_s, const float* lobe)
{
  // ローブ長が2画素の奇関数畳み込みカーネル
    return lobe[0] * (in_s[-2] - in_s[2])
	 + lobe[1] * (in_s[-1] - in_s[1]);    
}
    
/************************************************************************
*  __global__ functions							*
************************************************************************/
template <size_t L, class IN, class OUT> static __global__ void
fir_filterH_kernel(IN in, OUT out, int stride_i, int stride_o)
{
    constexpr size_t	LobeSize = L & ~0x1;	// 中心点を含まないローブ長

    int		xy = (blockIdx.y*blockDim.y + threadIdx.y)*stride_i
		   +  blockIdx.x*blockDim.x + threadIdx.x;
    const int	x  = LobeSize + threadIdx.x;
    const int	y  = threadIdx.y;

  // 原画像のブロックとその左右LobeSize分を共有メモリにコピー
    __shared__ float	in_s[BlockDimY][BlockDimX + 2*LobeSize + 1];
    in_s[y][x] = in[xy];
    if (threadIdx.x < LobeSize)
	in_s[y][x - LobeSize] = in[xy - LobeSize];
    if (threadIdx.x + LobeSize >= blockDim.x)
	in_s[y][x + LobeSize] = in[xy + LobeSize];
    __syncthreads();
    
  // 積和演算
    xy = (blockIdx.y*blockDim.y + threadIdx.y)*stride_o
       +  blockIdx.x*blockDim.x + threadIdx.x;
    out[xy] = convolve<L>(&in_s[y][x], lobeH());
}

template <size_t L, class IN, class OUT> static __global__ void
fir_filterV_kernel(const IN in, OUT out, int stride_i, int stride_o)
{
    constexpr size_t	LobeSize = L & ~0x1;	// 中心点を含まないローブ長

    int		xy = (blockIdx.y*blockDim.y + threadIdx.y)*stride_i
		   +  blockIdx.x*blockDim.x + threadIdx.x;
    const int	x  = threadIdx.x;
    const int	y  = LobeSize + threadIdx.y;

  // 原画像のブロックとその左右LobeSize分を共有メモリにコピー
    __shared__ float	in_s[BlockDimX][BlockDimY + 2*LobeSize + 1];
    in_s[x][y] = in[xy];
    if (threadIdx.y < LobeSize)
	in_s[x][y - LobeSize] = in[xy - LobeSize*stride_i];
    if (threadIdx.y + LobeSize >= blockDim.y) 
	in_s[x][y + LobeSize] = in[xy + LobeSize*stride_i];
    __syncthreads();
    
  // 積和演算
    xy = (blockIdx.y*blockDim.y + threadIdx.y)*stride_o
       +  blockIdx.x*blockDim.x + threadIdx.x;
    out[xy] = convolve<L>(&in_s[x][y], lobeV());
}

/************************************************************************
*  class FIRFilter2							*
************************************************************************/
template <size_t L, class IN, class OUT> void
FIRFilter2::convolveH(IN in, IN ie, OUT out)
{
    constexpr size_t	LobeSize = L & ~0x1;	// 中心点を含まないローブ長

    const auto	nrow = std::distance(in, ie);
    const auto	ncol = std::distance(in->begin(), in->end()) - 2*LobeSize;
    const auto	stride_i = stride(in);
    const auto	stride_o = stride(out);

  // 左上
    int		x = LobeSize;
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncol/threads.x, nrow/threads.y);
    fir_filterH_kernel<L><<<blocks, threads>>>(in->begin()  + x,
					       out->begin() + x,
					       stride_i, stride_o);
  // 右上
    x += blocks.x*threads.x;
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    fir_filterH_kernel<L><<<blocks, threads>>>(in->begin()  + x,
					       out->begin() + x,
					       stride_i, stride_o);
  // 左下
    std::advance(in,  blocks.y*threads.y);
    std::advance(out, blocks.y*threads.y);
    x = LobeSize;
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    threads.y = nrow%threads.y;
    blocks.y  = 1;
    fir_filterH_kernel<L><<<blocks, threads>>>(in->begin()  + x,
					       out->begin() + x,
					       stride_i, stride_o);
  // 右下
    x += blocks.x*threads.x;
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    fir_filterH_kernel<L><<<blocks, threads>>>(in->begin()  + x,
					       out->begin() + x,
					       stride_i, stride_o);
}

template <size_t L, class IN, class OUT> void
FIRFilter2::convolveV(IN in, IN ie, OUT out)
{
    constexpr size_t	LobeSize = L & ~0x1;	// 中心点を含まないローブ長

    const auto	nrow = std::distance(in, ie) - 2*LobeSize;
    const auto	ncol = std::distance(in->begin(), in->end());
    const auto	stride_i = stride(in);
    const auto	stride_o = stride(out);

  // 左上
    std::advance(in,  LobeSize);
    std::advance(out, LobeSize);
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncol/threads.x, nrow/threads.y);
    fir_filterV_kernel<L><<<blocks, threads>>>(in->begin(), out->begin(),
					       stride_i, stride_o);
  // 右上
    const int	x = blocks.x*threads.x;
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    fir_filterV_kernel<L><<<blocks, threads>>>(in->begin()  + x,
					       out->begin() + x,
					       stride_i, stride_o);
  // 左下
    std::advance(in,  blocks.y*threads.y);
    std::advance(out, blocks.y*threads.y);
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    threads.y = nrow%threads.y;
    blocks.y  = 1;
    fir_filterV_kernel<L><<<blocks, threads>>>(in->begin(), out->begin(),
					       stride_i, stride_o);
  // 右下
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    fir_filterV_kernel<L><<<blocks, threads>>>(in->begin()  + x,
					       out->begin() + x,
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

    const auto	ncol = std::distance(in->begin(), in->end());
    if (ncol < 4*(_lobeSizeH/2) + 1)
	return;
    
    _buf.resize(nrow, ncol);

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
#endif	// !__TU_CUDA_FIRFILTER_H
