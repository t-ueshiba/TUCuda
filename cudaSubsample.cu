/*
 * $Id: cudaSubsample.cu,v 1.3 2011-04-19 04:00:25 ueshiba Exp $
 */
#include "TU/CudaUtility.h"

namespace TU
{
/************************************************************************
*  global constatnt variables						*
************************************************************************/
static const size_t	BlockDimX = 32;
static const size_t	BlockDimY = 16;
    
/************************************************************************
*  device functions							*
************************************************************************/
template <class T> static __global__ void
subsample_kernel(const T* in, T* out, uint stride_i, uint stride_o)
{
    const int	tx = threadIdx.x,
		ty = threadIdx.y;
    const int	bw = blockDim.x;
    const int	x0 = blockIdx.x*bw,
		y  = blockIdx.y*blockDim.y + ty;
    const int	xy = 2*(y*stride_i + x0)   + tx;
    
  // 原画像の2x2ブロックを1行おきに共有メモリにコピー
    __shared__ T	in_s[BlockDimY][2*BlockDimX+1];
    in_s[ty][tx	    ] = in[xy     ];
    in_s[ty][tx + bw] = in[xy + bw];
    __syncthreads();

    out[y*stride_o + x0 + tx] = in_s[ty][2*tx];
}

/************************************************************************
*  global functions							*
************************************************************************/
//! CUDAによって2次元配列を水平／垂直方向それぞれ1/2に間引く．
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
*/
template <class T> void
cudaSubsample(const CudaArray2<T>& in, CudaArray2<T>& out)
{
    out.resize(in.nrow()/2, in.ncol()/2);

    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(out.ncol() / threads.x, out.nrow() / threads.y);

  // 左上
    subsample_kernel<<<blocks, threads>>>(in.data(), out.data(),
					  in.stride(), out.stride());

  // 下端
    uint	bottom = blocks.y * threads.y;
    threads.y = out.nrow() % threads.y;
    blocks.y  = 1;
    subsample_kernel<<<blocks, threads>>>(in.data()  + bottom * in.stride(),
					  out.data() + bottom * out.stride(),
					  in.stride(), out.stride());

  // 右端
    uint	right = blocks.x * threads.x;
    threads.x = out.ncol() % threads.x;
    blocks.x  = 1;
    threads.y = BlockDimY;
    blocks.y  = out.nrow() / threads.y;
    subsample_kernel<<<blocks, threads>>>(in.data()  + right,
					  out.data() + right,
					  in.stride(), out.stride());

  // 右下
    threads.y = out.nrow() % threads.y;
    blocks.y  = 1;
    subsample_kernel<<<blocks, threads>>>(in.data()  + bottom * in.stride()
						     + right,
					  out.data() + bottom * out.stride()
						     + right,
					  in.stride(), out.stride());
}

template void	cudaSubsample(const CudaArray2<u_char>& in,
				    CudaArray2<u_char>& out)	;
template void	cudaSubsample(const CudaArray2<short>& in,
				    CudaArray2<short>& out)	;
template void	cudaSubsample(const CudaArray2<int>& in,
				    CudaArray2<int>& out)	;
template void	cudaSubsample(const CudaArray2<float>& in,
				    CudaArray2<float>& out)	;
}
