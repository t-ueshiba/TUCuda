/*
 * $Id: FIRFilter.cu,v 1.7 2011-04-26 06:39:19 ueshiba Exp $
 */
/*!
  \file		FIRFilter.cu
*/
#include "TU/cuda/FIRFilter.h"
#include "TU/cuda/utility.h"

namespace TU
{
namespace cuda
{
/************************************************************************
*  global constatnt variables						*
************************************************************************/
static const size_t		BlockDimX = 32;
static const size_t		BlockDimY = 16;
    
static __constant__ float	_lobeH[FIRFilter2::LOBE_SIZE_MAX];
static __constant__ float	_lobeV[FIRFilter2::LOBE_SIZE_MAX];

/************************************************************************
*  device functions							*
************************************************************************/
static inline __device__ float
convolve(const float* in_s, const float* lobe,
	 std::integral_constant<size_t, 17>)
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
    
static inline __device__ float
convolve(const float* in_s, const float* lobe,
	 std::integral_constant<size_t, 16>)
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
    
static inline __device__ float
convolve(const float* in_s, const float* lobe,
	 std::integral_constant<size_t, 9>)
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
    
static inline __device__ float
convolve(const float* in_s, const float* lobe,
	 std::integral_constant<size_t, 8>)
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
    
static inline __device__ float
convolve(const float* in_s, const float* lobe,
	 std::integral_constant<size_t, 5>)
{
  // ローブ長が5画素の偶関数畳み込みカーネル
    return lobe[0] * (in_s[-4] + in_s[4])
	 + lobe[1] * (in_s[-3] + in_s[3])
	 + lobe[2] * (in_s[-2] + in_s[2])
	 + lobe[3] * (in_s[-1] + in_s[1])
	 + lobe[4] *  in_s[ 0];
}
    
static inline __device__ float
convolve(const float* in_s, const float* lobe,
	 std::integral_constant<size_t, 4>)
{
  // ローブ長が4画素の奇関数畳み込みカーネル
    return lobe[0] * (in_s[-4] - in_s[4])
	 + lobe[1] * (in_s[-3] - in_s[3])
	 + lobe[2] * (in_s[-2] - in_s[2])
	 + lobe[3] * (in_s[-1] - in_s[1]);
}
    
static inline __device__ float
convolve(const float* in_s, const float* lobe,
	 std::integral_constant<size_t, 3>)
{
  // ローブ長が3画素の偶関数畳み込みカーネル
    return lobe[0] * (in_s[-2] + in_s[2])
	 + lobe[1] * (in_s[-1] + in_s[1])
	 + lobe[2] *  in_s[ 0];
}
    
static inline __device__ float
convolve(const float* in_s, const float* lobe,
	 std::integral_constant<size_t, 2>)
{
  // ローブ長が2画素の奇関数畳み込みカーネル
    return lobe[0] * (in_s[-2] - in_s[2])
	 + lobe[1] * (in_s[-1] - in_s[1]);    
}
    
template <size_t L, class S, class T> static __global__ void
filterH_kernel(const S* in, T* out, uint stride_i, uint stride_o)
{
    const int	x   = blockIdx.x*blockDim.x + threadIdx.x,
		y   = blockIdx.y*blockDim.y + threadIdx.y,
		xy  = y*stride_i + x,
    		dxy = blockDim.x;

  // in_s[]を縦:blockDim.y, 横:3*blockDim.x の2次元配列として扱う．
    const int	xy_s  = threadIdx.y*(3*blockDim.x) + blockDim.x + threadIdx.x,
		dxy_s = blockDim.x;

  // 原画像の3つのタイル(スレッドブロックに対応)を共有メモリにコピー
    __shared__ float	in_s[BlockDimX * (3*BlockDimY + 1)];
    in_s[xy_s - dxy_s] = in[xy - dxy];
    in_s[xy_s	     ] = in[xy	    ];
    in_s[xy_s + dxy_s] = in[xy + dxy];
    __syncthreads();
    
  // 積和演算
    out[y*stride_o + x] = convolve(in_s + xy_s, _lobeH,
				   std::integral_constant<size_t, L>());
}
    
template <size_t L, class S, class T> static __global__ void
filterV_kernel(const S* in, T* out, uint stride_i, uint stride_o)
{
    const int	x   = blockIdx.x*blockDim.x + threadIdx.x,
		y   = blockIdx.y*blockDim.y + threadIdx.y,
		xy  = y*stride_i + x,
		dxy = blockDim.y*stride_i;

  // bank conflictを防ぐため，in_s[]を縦:blockDim.x, 横:3*blockDim.y + 1 の
  // 2次元配列として扱う．
    const int	xy_s  = threadIdx.x*(3*blockDim.y + 1)
		      + blockDim.y + threadIdx.y,
		dxy_s = blockDim.y;
    
  // 原画像の3つのタイル(スレッドブロックに対応)を共有メモリにコピー
    __shared__ float	in_s[BlockDimX * (3*BlockDimY + 1)];
    in_s[xy_s - dxy_s] = in[xy - dxy];
    in_s[xy_s	     ] = in[xy	    ];
    in_s[xy_s + dxy_s] = in[xy + dxy];
    __syncthreads();
    
  // 積和演算
    out[y*stride_o + x] = convolve(in_s + xy_s, _lobeH,
				   std::integral_constant<size_t, L>());
}

/************************************************************************
*  static functions							*
************************************************************************/
template <size_t L, class S, class T> inline static void
convolveH_dispatch(const CudaArray2<S>& in, CudaArray2<T>& out)
{
    const size_t	lobeSize = L & ~0x1;	// 中心点を含まないローブ長

  // 左上
    int		xs = lobeSize;
    dim3	threads(lobeSize, BlockDimY);
    dim3	blocks((BlockDimX - xs) / threads.x, 1);
    filterH_kernel<L><<<blocks, threads>>>(in[ 0].data().get() + xs,
					   out[0].data().get() + xs,
					   in.stride(), out.stride());
    xs += blocks.x * threads.x;

  // 右上
    threads.x = BlockDimX;
    blocks.x  = (out.stride() - xs) / threads.x;
    filterH_kernel<L><<<blocks, threads>>>(in[ 0].data().get() + xs,
					   out[0].data().get() + xs,
					   in.stride(), out.stride());
    int		ys = blocks.y * threads.y;
    if (ys >= in.nrow())
	return;

  // 中央
    blocks.x = out.stride() / threads.x;
    blocks.y = (out.nrow() - ys) / threads.y;
    filterH_kernel<L><<<blocks, threads>>>(in[ ys].data().get() + xs,
					   out[ys].data().get() + xs,
					   in.stride(), out.stride());
    ys += blocks.y * threads.y;
    if (ys >= in.nrow())
	return;

  // 左下
    blocks.x  = (out.stride() - lobeSize) / threads.x;
    threads.y = out.nrow() - ys;
    blocks.y  = 1;
    filterH_kernel<L><<<blocks, threads>>>(in[ ys].data().get(),
					   out[ys].data().get(),
					   in.stride(), out.stride());
    xs = blocks.x * threads.x;

  // 右下
    threads.x = lobeSize;
    blocks.x  = (out.stride() - lobeSize - xs) / threads.x;
    filterH_kernel<L><<<blocks, threads>>>(in[ ys].data().get() + xs,
					   out[ys].data().get() + xs,
					   in.stride(), out.stride());
}
    
template <size_t L, class S, class T> inline static void
convolveV_dispatch(const CudaArray2<S>& in, CudaArray2<T>& out)
{
    const size_t	lobeSize = L & ~0x1;	// 中心点を含まないローブ長

  // 最初のBlockDimY行（最初のlobeSize行は不定）
    int		ys = lobeSize;
    dim3	threads(BlockDimX, lobeSize);
    dim3	blocks(out.stride() / threads.x, (BlockDimY - ys) / threads.y);
    filterV_kernel<L><<<blocks, threads>>>(in[ ys].data().get(),
					   out[ys].data().get(),
					   in.stride(), out.stride());
    ys += blocks.y * threads.y;
    if (ys >= in.nrow())
	return;
    
  // BlockDimY行以上が残るように縦方向スレッド数をBlockDimYにして処理
    threads.y = BlockDimY;
    blocks.y  = (out.nrow() - ys) / threads.y - 1;
    filterV_kernel<L><<<blocks, threads>>>(in[ ys].data().get(),
					   out[ys].data().get(),
					   in.stride(), out.stride());
    ys += blocks.y * threads.y;
    if (ys >= in.nrow())
	return;
    
  // 残りは縦方向スレッド数をlobeSizeにして処理（最後のlobeSize行は不定）
    threads.y = lobeSize;
    blocks.y  = (out.nrow() - ys - 1) / threads.y;
    ys = out.nrow() - (1 + blocks.y) * threads.y;
    filterV_kernel<L><<<blocks, threads>>>(in[ ys].data().get(),
					   out[ys].data().get(),
					   in.stride(), out.stride());
}
    
/************************************************************************
*  class FIRFilter2							*
************************************************************************/
//! CUDAによる2次元フィルタを生成する．
FIRFilter2::FIRFilter2()
    :_lobeSizeH(0), _lobeSizeV(0)
{
    int	device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&_prop, device);
}

//! 2次元フィルタのローブを設定する．
/*!
  与えるローブの長さは，畳み込みカーネルが偶関数の場合2^n + 1, 奇関数の場合2^n
  (n = 1, 2, 3, 4)でなければならない．
  \param lobeH	横方向ローブ
  \param lobeV	縦方向ローブ
  \return	この2次元フィルタ
*/
FIRFilter2&
FIRFilter2::initialize(const Array<float>& lobeH, const Array<float>& lobeV)
{
    using namespace	std;
    
    if (lobeH.size() > LOBE_SIZE_MAX || lobeV.size() > LOBE_SIZE_MAX)
	throw runtime_error("FIRFilter2::initialize: too large lobe size!");
    
    _lobeSizeH = lobeH.size();
    _lobeSizeV = lobeV.size();
    cudaMemcpyToSymbol(_lobeH, lobeH.data(), lobeH.size()*sizeof(float));
    cudaMemcpyToSymbol(_lobeV, lobeV.data(), lobeV.size()*sizeof(float));

    return *this;
}
    
//! 与えられた2次元配列とこのフィルタを畳み込む
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このフィルタ自身
*/
template <class S, class T> const FIRFilter2&
FIRFilter2::convolve(const CudaArray2<S>& in, CudaArray2<T>& out) const
{
    using namespace	std;

  // 横方向に畳み込む．
    _buf.resize(in.nrow(), in.ncol());

    switch (_lobeSizeH)
    {
      case 17:
	convolveH_dispatch<17>(in, _buf);
	break;
      case 16:
	convolveH_dispatch<16>(in, _buf);
	break;
      case  9:
	convolveH_dispatch< 9>(in, _buf);
	break;
      case  8:
	convolveH_dispatch< 9>(in, _buf);
	break;
      case  5:
	convolveH_dispatch< 5>(in, _buf);
	break;
      case  4:
	convolveH_dispatch< 4>(in, _buf);
	break;
      case  3:
	convolveH_dispatch< 3>(in, _buf);
	break;
      case  2:
	convolveH_dispatch< 2>(in, _buf);
	break;
      default:
	throw runtime_error("FIRFilter2::convolve: unsupported horizontal lobe size!");
    }

  // 縦方向に畳み込む．
    out.resize(_buf.nrow(), _buf.ncol());
    
    switch (_lobeSizeV)
    {
      case 17:
	convolveV_dispatch<17>(_buf, out);
	break;
      case 16:
	convolveV_dispatch<16>(_buf, out);
	break;
      case  9:
	convolveV_dispatch< 9>(_buf, out);
	break;
      case  8:
	convolveV_dispatch< 8>(_buf, out);
	break;
      case  5:
	convolveV_dispatch< 5>(_buf, out);
	break;
      case  4:
	convolveV_dispatch< 4>(_buf, out);
	break;
      case  3:
	convolveV_dispatch< 3>(_buf, out);
	break;
      case  2:
	convolveV_dispatch< 2>(_buf, out);
	break;
      default:
	throw runtime_error("FIRFilter2::convolve: unsupported vertical lobe size!");
    }

    return *this;
}

template const FIRFilter2&
FIRFilter2::convolve(const CudaArray2<u_char>& in,
			   CudaArray2<u_char>& out)		const	;
template const FIRFilter2&
FIRFilter2::convolve(const CudaArray2<u_char>& in,
			   CudaArray2<float>&  out)		const	;
template const FIRFilter2&
FIRFilter2::convolve(const CudaArray2<float>& in,
			   CudaArray2<u_char>& out)		const	;
template const FIRFilter2&
FIRFilter2::convolve(const CudaArray2<float>& in,
			   CudaArray2<float>& out)		const	;
}
}
