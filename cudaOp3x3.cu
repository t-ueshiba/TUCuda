/*
 *  $Id: cudaOp3x3.cu,v 1.1 2011-04-18 08:16:55 ueshiba Exp $
 */
#include "TU/CudaUtility.h"

namespace TU
{
/************************************************************************
*  global constatnt variables						*
************************************************************************/
static const int	BlockDim = 16;	// u_intにするとCUDAのバグを踏む！
    
/************************************************************************
*  device functions							*
************************************************************************/
template <class T, class OP> static __global__ void
op3x3_kernel(const T* in, float* out, u_int stride_i, u_int stride_o, OP op3x3)
{
  // このカーネルはブロック境界処理のために blockDim.x == blockDim.y を仮定
    int	lft = (blockIdx.y * blockDim.y + threadIdx.y) * stride_i 
	    +  blockIdx.x * blockDim.x,			// 現在位置から見て左端
	xy  = lft + threadIdx.x;			// 現在位置
    int	x_s = threadIdx.x + 1,
	y_s = threadIdx.y + 1;

  // 原画像のブロック内部およびその外枠1画素分を共有メモリに転送
    __shared__ float	in_s[BlockDim+2][BlockDim+2];

    in_s[y_s][x_s] = in[xy];				// 内部

    if (threadIdx.y == 0)	// ブロックの上端?
    {

	int	top = xy - stride_i;			// 現在位置の直上
	int	bot = xy + blockDim.y * stride_i;	// 現在位置から見て下端
	in_s[	      0][x_s] = in[top];		// 上枠
	in_s[BlockDim+1][x_s] = in[bot];		// 下枠

	lft += threadIdx.x * stride_i;
	in_s[x_s][	   0] = in[lft - 1];		// 左枠
	in_s[x_s][BlockDim+1] = in[lft + BlockDim];	// 右枠

	if (threadIdx.x == 0)	// ブロックの左上隅?
	{
	    if ((blockIdx.x != 0) || (blockIdx.y != 0))
	  	in_s[0][0] = in[top - 1];			    // 左上隅
	    if ((blockIdx.x != gridDim.x - 1) || (blockIdx.y != gridDim.y - 1))
		in_s[BlockDim+1][BlockDim+1] = in[bot + BlockDim];  // 右下隅
	    in_s[0][BlockDim+1] = in[top + BlockDim];		    // 右上隅
	    in_s[BlockDim+1][0] = in[bot - 1];			    // 左下隅
	}
    }
    __syncthreads();

  // 共有メモリに保存した原画像データから現在画素に対するフィルタ出力を計算
    int	xy_o = (blockIdx.y * blockDim.y + threadIdx.y) * stride_o
	     +  blockIdx.x * blockDim.x + threadIdx.x;
    --x_s;
    out[xy_o] = op3x3(in_s[y_s-1] + x_s, in_s[y_s] + x_s, in_s[y_s+1] + x_s);
}

/************************************************************************
*  global functions							*
************************************************************************/
template <class T, class OP> inline static void
cudaOp3x3(const CudaArray2<T>& in, CudaArray2<float>& out, OP op)
{
    using namespace	std;
    
    if (in.nrow() < 3 || in.ncol() < 3)
	return;
    
    out.resize(in.nrow(), in.ncol());

  // 最初と最後の行を除いた (in.nrow() - 2) x in.ncol() の配列として扱う
    dim3	threads(BlockDim, BlockDim);
    dim3	blocks(in.ncol()/threads.x, (in.nrow() - 2)/threads.y);
    
  // 左上
    op3x3_kernel<<<blocks, threads>>>((const T*)in[1], (float*)out[1],
				      in.stride(), out.stride(), op);
#ifndef NO_BORDER
  /*  // 右上
    uint	offset = threads.y * blocks.y;
    threads.x = threads.y = in.ncol() - offset;
    blocks.x = 1;
    blocks.y = (in.nrow() - 2)/threads.y;
    op3x3_kernel<<<blocks, threads>>>((const T*)in[1]  + offset,
				      (  float*)out[1] + offset,
				      in.stride(), out.stride(), op);


    dim3	threads(BlockDim, BlockDim);
    for (int i = 2; i < in.nrow(); )
    {
	threads.x = threads.y = std::min(in.nrow() - i, in.ncol() - j);
	blocks.y = (in.nrow() - i) / threads.y;
	


	i += threads.y * blocks.y;
	}*/
#endif
}

template void
cudaOp3x3(const CudaArray2<u_char>& in,
		CudaArray2<float>& out, laplacian3x3<float> op)		;
template void
cudaOp3x3(const CudaArray2<float>& in,
		CudaArray2<float>& out, laplacian3x3<float> op)		;
template void
cudaOp3x3(const CudaArray2<u_char>& in,
		CudaArray2<float>& out, det3x3<float> op)		;
template void
cudaOp3x3(const CudaArray2<float>& in,
		CudaArray2<float>& out, det3x3<float> op)		;

}
