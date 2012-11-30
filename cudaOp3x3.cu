/*
 *  $Id: cudaOp3x3.cu,v 1.7 2011-05-09 00:35:49 ueshiba Exp $
 */
#include "TU/CudaUtility.h"

namespace TU
{
/************************************************************************
*  global constatnt variables						*
************************************************************************/
static const u_int	BlockDim = 16;		// ブロックサイズの初期値
    
/************************************************************************
*  device functions							*
************************************************************************/
template <class S, class T, class OP> static __global__ void
op3x3_kernel(const S* in, T* out, u_int stride_i, u_int stride_o, OP op)
{
  // このカーネルはブロック境界処理のために blockDim.x == blockDim.y を仮定
    const int	blk = blockDim.x;	// u_intにするとダメ．CUDAのバグ？
    int		xy  = (blockIdx.y * blk + threadIdx.y) * stride_i 
		    +  blockIdx.x * blk + threadIdx.x;	// 現在位置
    int		x   = 1 + threadIdx.x;
    const int	y   = 1 + threadIdx.y;

  // 原画像のブロック内部およびその外枠1画素分を共有メモリに転送
    __shared__ S	in_s[BlockDim+2][BlockDim+2];
    in_s[y][x] = in[xy];				// 内部

    if (threadIdx.y == 0)	// ブロックの上端?
    {
	const int	top = xy - stride_i;		// 現在位置の直上
	const int	bot = xy + blk * stride_i;	// 現在位置の下端
	in_s[	   0][x] = in[top];			// 上枠
	in_s[blk + 1][x] = in[bot];			// 下枠

	const int	lft = xy + threadIdx.x * (stride_i - 1);
	in_s[x][      0] = in[lft -   1];		// 左枠
	in_s[x][blk + 1] = in[lft + blk];		// 右枠

	if (threadIdx.x == 0)	// ブロックの左上隅?
	{
	    if ((blockIdx.x != 0) || (blockIdx.y != 0))
	  	in_s[0][0] = in[top - 1];		// 左上隅
	    if ((blockIdx.x != gridDim.x - 1) || (blockIdx.y != gridDim.y - 1))
		in_s[blk + 1][blk + 1] = in[bot + blk];	// 右下隅
	    in_s[0][blk + 1] = in[top + blk];		// 右上隅
	    in_s[blk + 1][0] = in[bot -   1];		// 左下隅
	}
    }
    __syncthreads();

  // 共有メモリに保存した原画像データから現在画素に対するフィルタ出力を計算
    xy = (blockIdx.y * blk + threadIdx.y) * stride_o
       +  blockIdx.x * blk + threadIdx.x;
    --x;
    out[xy] = op(in_s[y - 1] + x, in_s[y] + x, in_s[y + 1] + x);
}

/************************************************************************
*  global functions							*
************************************************************************/
//! CUDAによって2次元配列に対して3x3近傍演算を行う．
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \param op	3x3近傍演算子
*/
template <class S, class T, class OP> void
cudaOp3x3(const CudaArray2<S>& in, CudaArray2<T>& out, OP op)
{
    using namespace	std;
    
    if (in.nrow() < 3 || in.ncol() < 3)
	return;
    
    out.resize(in.nrow(), in.ncol());

  // 最初と最後の行を除いた (out.nrow() - 2) x out.stride() の配列として扱う
    dim3	threads(BlockDim, BlockDim);
    dim3	blocks(out.stride()/threads.x, (out.nrow() - 2)/threads.y);
    op3x3_kernel<<<blocks, threads>>>(in[1].ptr(), out[1].ptr(),
				      in.stride(), out.stride(), op);

  // 左下
    int	top = 1 + threads.y * blocks.y;
    threads.x = threads.y = out.nrow() - top - 1;
    if (threads.x == 0)
	return;
    blocks.x = out.stride() / threads.x;
    blocks.y = 1;
    op3x3_kernel<<<blocks, threads>>>(in[top].ptr(), out[top].ptr(),
				      in.stride(), out.stride(), op);

  // 右下
    if (threads.x * blocks.x == out.stride())
	return;
    int	lft = out.stride() - threads.x;
    blocks.x = 1;
    op3x3_kernel<<<blocks, threads>>>(in[top].ptr()  + lft,
				      out[top].ptr() + lft,
				      in.stride(), out.stride(), op);
}

template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  diffH3x3<u_char, float> op)					;
template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  diffH3x3<float, float> op)					;

template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  diffV3x3<u_char, float> op)					;
template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  diffV3x3<float, float> op)					;

template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  diffHH3x3<u_char, float> op)					;
template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  diffHH3x3<float, float> op)					;

template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  diffVV3x3<u_char, float> op)					;
template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  diffVV3x3<float, float> op)					;

template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  diffHV3x3<float, float> op)					;
template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  diffHV3x3<u_char, float> op)					;

template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  sobelH3x3<u_char, float> op)					;
template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  sobelH3x3<float, float> op)					;

template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  sobelV3x3<u_char, float> op)					;
template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  sobelV3x3<float, float> op)					;

template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  sobelAbs3x3<u_char, float> op)				;
template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  sobelAbs3x3<float, float> op)					;

template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  laplacian3x3<u_char, float> op)				;
template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  laplacian3x3<float, float> op)				;

template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  det3x3<float, float> op)					;
template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  det3x3<u_char, float> op)					;

template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  maximal3x3<float> op)						;

template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  minimal3x3<float> op)						;
}
