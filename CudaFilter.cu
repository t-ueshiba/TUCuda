/*
 * $Id: CudaFilter.cu,v 1.5 2011-04-20 08:15:07 ueshiba Exp $
 */
#include "TU/CudaFilter.h"
#include "TU/CudaUtility.h"

namespace TU
{
/************************************************************************
*  global constatnt variables						*
************************************************************************/
static const u_int		BlockDimX = 32;
static const u_int		BlockDimY = 16;
    
static __constant__ float	_lobeH[CudaFilter2::LOBE_SIZE_MAX];
static __constant__ float	_lobeV[CudaFilter2::LOBE_SIZE_MAX];

/************************************************************************
*  device functions							*
************************************************************************/
template <uint D, class S, class T> static __global__ void
filter_kernel(const S* in, T* out, uint stride_i, uint stride_o)
{
    extern __shared__ float	in_s[];
    int				xy, dxy, xy_s, dxy_s, xy_o;
    const float*		lobe;
    
  // D = 2*ローブ長 (横方向）または D = 2*ローブ長 + 1（縦方向）
    if (D & 0x1)	// 縦方向にフィルタリング
    {
	int	x = (blockIdx.x + 1)*blockDim.x + threadIdx.x,
		y = (blockIdx.y + 1)*blockDim.y + threadIdx.y;

	xy    = y*stride_i + x;
	dxy   = blockDim.y*stride_i;
	
	xy_o  = y*stride_o + x;
	
      // bank conflictを防ぐため，in_s[]を縦:blockDim.x, 横:3*blockDim.y + 1 の
      // 2次元配列として扱う．
	xy_s  = threadIdx.x*(3*blockDim.y + 1) + blockDim.y + threadIdx.y;
	dxy_s = blockDim.y;

	lobe = _lobeV;
    }
    else		// 横方向にフィルタリング
    {
	int	x = (blockIdx.x + 1)*blockDim.x + threadIdx.x,
		y = blockIdx.y	    *blockDim.y + threadIdx.y;

	xy    = y*stride_i + x;
	dxy   = blockDim.x;

	xy_o  = y*stride_o + x;

      // in_s[]を縦:blockDim.y, 横:3*blockDim.x の2次元配列として扱う．
    	xy_s  = threadIdx.y*(3*blockDim.x) + blockDim.x + threadIdx.x;
	dxy_s = blockDim.x;

	lobe = _lobeH;
    }
    
  // 原画像の3つのタイル(スレッドブロックに対応)を共有メモリにコピー
    in_s[xy_s - dxy_s] = in[xy - dxy];
    in_s[xy_s	     ] = in[xy	    ];
    in_s[xy_s + dxy_s] = in[xy + dxy];
    __syncthreads();
    
  // 積和演算
    switch (D >> 1)
    {
      case 17:	// ローブ長が17画素の偶関数畳み込みカーネル
	out[xy_o] = lobe[ 0] * (in_s[xy_s - 16] + in_s[xy_s + 16])
		  + lobe[ 1] * (in_s[xy_s - 15] + in_s[xy_s + 15])
		  + lobe[ 2] * (in_s[xy_s - 14] + in_s[xy_s + 14])
		  + lobe[ 3] * (in_s[xy_s - 13] + in_s[xy_s + 13])
		  + lobe[ 4] * (in_s[xy_s - 12] + in_s[xy_s + 12])
		  + lobe[ 5] * (in_s[xy_s - 11] + in_s[xy_s + 11])
		  + lobe[ 6] * (in_s[xy_s - 10] + in_s[xy_s + 10])
		  + lobe[ 7] * (in_s[xy_s -  9] + in_s[xy_s +  9])
		  + lobe[ 8] * (in_s[xy_s -  8] + in_s[xy_s +  8])
		  + lobe[ 9] * (in_s[xy_s -  7] + in_s[xy_s +  7])
		  + lobe[10] * (in_s[xy_s -  6] + in_s[xy_s +  6])
		  + lobe[11] * (in_s[xy_s -  5] + in_s[xy_s +  5])
		  + lobe[12] * (in_s[xy_s -  4] + in_s[xy_s +  4])
		  + lobe[13] * (in_s[xy_s -  3] + in_s[xy_s +  3])
		  + lobe[14] * (in_s[xy_s -  2] + in_s[xy_s +  2])
		  + lobe[15] * (in_s[xy_s -  1] + in_s[xy_s +  1])
		  + lobe[16] *  in_s[xy_s     ];
	break;
      case 16:	// ローブ長が16画素の奇関数畳み込みカーネル
	out[xy_o] = lobe[ 0] * (in_s[xy_s - 16] - in_s[xy_s + 16])
		  + lobe[ 1] * (in_s[xy_s - 15] - in_s[xy_s + 15])
		  + lobe[ 2] * (in_s[xy_s - 14] - in_s[xy_s + 14])
		  + lobe[ 3] * (in_s[xy_s - 13] - in_s[xy_s + 13])
		  + lobe[ 4] * (in_s[xy_s - 12] - in_s[xy_s + 12])
		  + lobe[ 5] * (in_s[xy_s - 11] - in_s[xy_s + 11])
		  + lobe[ 6] * (in_s[xy_s - 10] - in_s[xy_s + 10])
		  + lobe[ 7] * (in_s[xy_s -  9] - in_s[xy_s +  9])
		  + lobe[ 8] * (in_s[xy_s -  8] - in_s[xy_s +  8])
		  + lobe[ 9] * (in_s[xy_s -  7] - in_s[xy_s +  7])
		  + lobe[10] * (in_s[xy_s -  6] - in_s[xy_s +  6])
		  + lobe[11] * (in_s[xy_s -  5] - in_s[xy_s +  5])
		  + lobe[12] * (in_s[xy_s -  4] - in_s[xy_s +  4])
		  + lobe[13] * (in_s[xy_s -  3] - in_s[xy_s +  3])
		  + lobe[14] * (in_s[xy_s -  2] - in_s[xy_s +  2])
		  + lobe[15] * (in_s[xy_s -  1] - in_s[xy_s +  1]);
	break;
      case  9:	// ローブ長が9画素の偶関数畳み込みカーネル
	out[xy_o] = lobe[ 0] * (in_s[xy_s -  8] + in_s[xy_s +  8])
		  + lobe[ 1] * (in_s[xy_s -  7] + in_s[xy_s +  7])
		  + lobe[ 2] * (in_s[xy_s -  6] + in_s[xy_s +  6])
		  + lobe[ 3] * (in_s[xy_s -  5] + in_s[xy_s +  5])
		  + lobe[ 4] * (in_s[xy_s -  4] + in_s[xy_s +  4])
		  + lobe[ 5] * (in_s[xy_s -  3] + in_s[xy_s +  3])
		  + lobe[ 6] * (in_s[xy_s -  2] + in_s[xy_s +  2])
		  + lobe[ 7] * (in_s[xy_s -  1] + in_s[xy_s +  1])
		  + lobe[ 8] *  in_s[xy_s     ];
	break;
      case  8:	// ローブ長が8画素の奇関数畳み込みカーネル
	out[xy_o] = lobe[ 0] * (in_s[xy_s -  8] - in_s[xy_s +  8])
		  + lobe[ 1] * (in_s[xy_s -  7] - in_s[xy_s +  7])
		  + lobe[ 2] * (in_s[xy_s -  6] - in_s[xy_s +  6])
		  + lobe[ 3] * (in_s[xy_s -  5] - in_s[xy_s +  5])
		  + lobe[ 4] * (in_s[xy_s -  4] - in_s[xy_s +  4])
		  + lobe[ 5] * (in_s[xy_s -  3] - in_s[xy_s +  3])
		  + lobe[ 6] * (in_s[xy_s -  2] - in_s[xy_s +  2])
		  + lobe[ 7] * (in_s[xy_s -  1] - in_s[xy_s +  1]);
	break;
      case  5:	// ローブ長が5画素の偶関数畳み込みカーネル
	out[xy_o] = lobe[0] * (in_s[xy_s - 4] + in_s[xy_s + 4])
		  + lobe[1] * (in_s[xy_s - 3] + in_s[xy_s + 3])
		  + lobe[2] * (in_s[xy_s - 2] + in_s[xy_s + 2])
		  + lobe[3] * (in_s[xy_s - 1] + in_s[xy_s + 1])
		  + lobe[4] *  in_s[xy_s    ];
	break;
      case  4:	// ローブ長が4画素の奇関数畳み込みカーネル
	out[xy_o] = lobe[0] * (in_s[xy_s - 4] - in_s[xy_s + 4])
		  + lobe[1] * (in_s[xy_s - 3] - in_s[xy_s + 3])
		  + lobe[2] * (in_s[xy_s - 2] - in_s[xy_s + 2])
		  + lobe[3] * (in_s[xy_s - 1] - in_s[xy_s + 1]);
	break;
      case  3:	// ローブ長が2画素の偶関数畳み込みカーネル
	out[xy_o] = lobe[0] * (in_s[xy_s - 2] + in_s[xy_s + 2])
		  + lobe[1] * (in_s[xy_s - 1] + in_s[xy_s + 1])
		  + lobe[2] *  in_s[xy_s    ];
	break;
      case  2:	// ローブ長が2画素の奇関数畳み込みカーネル
	out[xy_o] = lobe[0] * (in_s[xy_s - 2] - in_s[xy_s + 2])
		  + lobe[1] * (in_s[xy_s - 1] - in_s[xy_s + 1]);
	break;
    }
}

/************************************************************************
*  static functions							*
************************************************************************/
template <uint D, class S, class T> inline static void
convolve_dispatch(const CudaArray2<S>& in, CudaArray2<T>& out)
{
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(in.ncol()/threads.x - 2,
		       in.nrow()/threads.y - 2*(D & 0x1));
    uint	shmsize = threads.x*(3*threads.y + (D & 0x1))*sizeof(float);

  // 中心部
    filter_kernel<D><<<blocks, threads, shmsize>>>((const S*)in, (T*)out,
						   in.stride(),
						   out.stride());
#ifndef NO_BORDER
  // 左端と右端
    const uint	lobeSize = (D >> 1) & ~0x1;	// 中心点を含まないローブ長
    uint	offset = (1 + blocks.x)*threads.x - lobeSize;
    blocks.x  = threads.x/lobeSize - 1;
    threads.x = lobeSize;
    filter_kernel<D><<<blocks, threads, shmsize>>>((const S*)in, (T*)out,
						   in.stride(),
						   out.stride());
    filter_kernel<D><<<blocks, threads, shmsize>>>((const S*)in  + offset,
						   (	  T*)out + offset,
						   in.stride(),
						   out.stride());

    if (D & 0x1)	// 縦方向にフィルタリング
    {
      // 上端と下端
	offset	  = (1 + blocks.y)*threads.y - lobeSize;
	blocks.x  = in.ncol()/threads.x - 2;
	blocks.y  = threads.y/lobeSize  - 1;
	threads.y = lobeSize;
	filter_kernel<D><<<blocks, threads, shmsize>>>((const S*)in, (T*)out,
						       in.stride(),
						       out.stride());
	filter_kernel<D><<<blocks, threads, shmsize>>>(
	    (const S*)in  + offset*in.stride(),
	    (	   T*)out + offset*out.stride(),
	    in.stride(), out.stride());
    }
#endif
}
    
/************************************************************************
*  class CudaFilter2							*
************************************************************************/
//! CUDAによる2次元フィルタを生成する．
CudaFilter2::CudaFilter2()
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
CudaFilter2&
CudaFilter2::initialize(const Array<float>& lobeH, const Array<float>& lobeV)
{
    using namespace	std;
    
    if (_lobeSizeH > LOBE_SIZE_MAX || _lobeSizeV > LOBE_SIZE_MAX)
	throw runtime_error("CudaFilter2::initialize: too large lobe size!");
    
    _lobeSizeH = lobeH.size();
    _lobeSizeV = lobeV.size();
    cudaCopyToConstantMemory(lobeH.begin(), lobeH.end(), _lobeH);
    cudaCopyToConstantMemory(lobeV.begin(), lobeV.end(), _lobeV);

    return *this;
}
    
//! 与えられた2次元配列とこのフィルタを畳み込む
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このフィルタ自身
*/
template <class S, class T> const CudaFilter2&
CudaFilter2::convolve(const CudaArray2<S>& in, CudaArray2<T>& out) const
{
    convolveH(in, _buf);
    convolveV(_buf, out);

    return *this;
}
    
//! 与えられた2次元配列とこのフィルタを横方向に畳み込む
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このフィルタ自身
*/
template <class S, class T> const CudaFilter2&
CudaFilter2::convolveH(const CudaArray2<S>& in, CudaArray2<T>& out) const
{
    using namespace	std;
    
    out.resize(in.nrow(), in.ncol());

    switch (_lobeSizeH)
    {
      case 17:
	convolve_dispatch<34>(in, out);
	break;
      case 16:
	convolve_dispatch<32>(in, out);
	break;
      case  9:
	convolve_dispatch<18>(in, out);
	break;
      case  8:
	convolve_dispatch<16>(in, out);
	break;
      case  5:
	convolve_dispatch<10>(in, out);
	break;
      case  4:
	convolve_dispatch< 8>(in, out);
	break;
      case  3:
	convolve_dispatch< 6>(in, out);
	break;
      case  2:
	convolve_dispatch< 4>(in, out);
	break;
      default:
	throw runtime_error("CudaFilter2::convolveH: unsupported horizontal lobe size!");
    }

    return *this;
}

//! 与えられた2次元配列とこのフィルタを縦方向に畳み込む
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \return	このフィルタ自身
*/
template <class S, class T> const CudaFilter2&
CudaFilter2::convolveV(const CudaArray2<S>& in, CudaArray2<T>& out) const
{
    using namespace	std;
    
    out.resize(in.nrow(), in.ncol());
    
    switch (_lobeSizeV)
    {
      case 17:
	convolve_dispatch<35>(in, out);
	break;
      case 16:
	convolve_dispatch<33>(in, out);
	break;
      case  9:
	convolve_dispatch<19>(in, out);
	break;
      case  8:
	convolve_dispatch<17>(in, out);
	break;
      case  5:
	convolve_dispatch<11>(in, out);
	break;
      case  4:
	convolve_dispatch< 9>(in, out);
	break;
      case  3:
	convolve_dispatch< 7>(in, out);
	break;
      case  2:
	convolve_dispatch< 5>(in, out);
	break;
      default:
	throw runtime_error("CudaFilter2::convolveV: unsupported vertical lobe size!");
    }

    return *this;
}

template const CudaFilter2&
CudaFilter2::convolve(const CudaArray2<u_char>& in,
			    CudaArray2<u_char>& out)		const	;
template const CudaFilter2&
CudaFilter2::convolve(const CudaArray2<u_char>& in,
			    CudaArray2<float>&  out)		const	;
template const CudaFilter2&
CudaFilter2::convolve(const CudaArray2<float>& in,
			    CudaArray2<u_char>& out)		const	;
template const CudaFilter2&
CudaFilter2::convolve(const CudaArray2<float>& in,
			    CudaArray2<float>& out)		const	;
}
