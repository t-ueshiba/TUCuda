/*
 *  $Id: cudaSuppressNonExtrema3x3.cu,v 1.1 2011-04-28 07:59:04 ueshiba Exp $
 */
#include "TU/CudaUtility.h"
#include <thrust/functional.h>

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
extrema3x3_kernel(const S* in, T* out,
		  u_int stride_i, u_int stride_o, OP op, T nulval)
{
  // このカーネルはブロック境界処理のために blockDim.x == blockDim.y を仮定
    const int	blk = blockDim.x;	// u_intにするとダメ．CUDAのバグ？
    int		xy  = 2*(blockIdx.y*blockDim.y + threadIdx.y)*stride_i
		    + 2*blockIdx.x*blockDim.y + threadIdx.x;
    int		x_s = 1 +   threadIdx.x;
    int		y_s = 1 + 2*threadIdx.y;

  // (2*blockDim.x)x(2*blockDim.y) の矩形領域を入力用共有メモリ領域にコピー
    __shared__ S	in_s[2*BlockDim + 2][2*BlockDim + 3];
    in_s[y_s    ][x_s	   ] = in[xy		     ];
    in_s[y_s    ][x_s + blk] = in[xy		+ blk];
    in_s[y_s + 1][x_s	   ] = in[xy + stride_i	     ];
    in_s[y_s + 1][x_s + blk] = in[xy + stride_i + blk];

  // 2x2ブロックの外枠を入力用共有メモリ領域にコピー
    if (threadIdx.y == 0)	// ブロックの上端?
    {
	const int	blk2 = 2*blockDim.y;
	const int	top  = xy - stride_i;		// 現在位置の直上
	const int	bot  = xy + blk2 * stride_i;	// 現在位置の下端
	in_s[0	     ][x_s	] = in[top	];	// 上枠左半
	in_s[0	     ][x_s + blk] = in[top + blk];	// 上枠右半
	in_s[1 + blk2][x_s	] = in[bot	];	// 下枠左半
	in_s[1 + blk2][x_s + blk] = in[bot + blk];	// 下枠右半

	int	lft = xy + threadIdx.x*(stride_i - 1);
	in_s[x_s      ][0	] = in[lft -	1];	// 左枠上半
	in_s[x_s      ][1 + blk2] = in[lft + blk2];	// 右枠上半
	lft += blockDim.y * stride_i;
	in_s[x_s + blk][0	] = in[lft -	1];	// 左枠下半
	in_s[x_s + blk][1 + blk2] = in[lft + blk2];	// 右枠下半

	if (threadIdx.x == 0)	// ブロックの左上隅?
	{
	    if ((blockIdx.x != 0) || (blockIdx.y != 0))
	  	in_s[0][0] = in[top - 1];			// 左上隅
	    if ((blockIdx.x != gridDim.x - 1) || (blockIdx.y != gridDim.y - 1))
		in_s[1 + blk2][1 + blk2] = in[bot + blk2];	// 右下隅
	    in_s[0][1 + blk2] = in[top + blk2];			// 右上隅
	    in_s[1 + blk2][0] = in[bot -    1];			// 左下隅
	}
    }
    __syncthreads();

  // このスレッドの処理対象である2x2画素ウィンドウに非極値を表す値を書き込む．
    __shared__ T	out_s[2*BlockDim][2*BlockDim+1];
    x_s = 1 + 2*threadIdx.x;
    out_s[y_s - 1][x_s - 1] = out_s[y_s - 1][x_s]
			    = out_s[y_s][x_s - 1]
			    = out_s[y_s][x_s	] = nulval;

  // この2x2ウィンドウ中で最大/最小となる画素の座標を求める．
    const int	i01 = (op(in_s[y_s    ][x_s], in_s[y_s	  ][x_s + 1]) ? 0 : 1);
    const int	i23 = (op(in_s[y_s + 1][x_s], in_s[y_s + 1][x_s + 1]) ? 2 : 3);
    const int	iex = (op(in_s[y_s    ][x_s + i01],
			  in_s[y_s + 1][x_s + (i23 & 0x1)]) ? i01 : i23);
    x_s += (iex & 0x1);
    y_s += (iex >> 1);
    
    const T	val = in_s[y_s][x_s];
    const int	dx  = (iex & 0x1 ? 1 : -1);
    const int	dy  = (iex & 0x2 ? 1 : -1);
    out_s[y_s-1][x_s-1] = (op(val, in_s[y_s + dy][x_s - dx]) &&
			   op(val, in_s[y_s + dy][x_s	  ]) &&
			   op(val, in_s[y_s + dy][x_s + dx]) &&
			   op(val, in_s[y_s     ][x_s + dx]) &&
			   op(val, in_s[y_s - dy][x_s + dx]) ? val : nulval);
    __syncthreads();

  // (2*blockDim.x)x(2*blockDim.y) の矩形領域に出力用共有メモリ領域をコピー．
    x_s =   threadIdx.x;
    y_s = 2*threadIdx.y;
    xy  = (2*blockIdx.y*blockDim.y + y_s)*stride_o
	+  2*blockIdx.x*blockDim.y + x_s;
    out[xy		   ] = out_s[y_s    ][x_s      ];
    out[xy	      + blk] = out_s[y_s    ][x_s + blk];
    out[xy + stride_o	   ] = out_s[y_s + 1][x_s      ];
    out[xy + stride_o + blk] = out_s[y_s + 1][x_s + blk];
}
    
/************************************************************************
*  global functions							*
************************************************************************/
//! CUDAによって2次元配列に対して3x3非極値抑制処理を行う．
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
*/
template <class S, class T, class OP> void
cudaSuppressNonExtrema3x3(const CudaArray2<S>& in,
				CudaArray2<T>& out, OP op, T nulval)
{
    using namespace	std;
    
    if (in.nrow() < 3 || in.ncol() < 3)
	return;
    
    out.resize(in.nrow(), in.ncol());

  // 最初と最後の行を除いた (out.nrow() - 2) x out.stride() の配列として扱う
    dim3	threads(BlockDim, BlockDim);
    dim3	blocks(out.stride()     / (2*threads.x),
		       (out.nrow() - 2) / (2*threads.y));
    extrema3x3_kernel<<<blocks, threads>>>((const S*)in[1], (T*)out[1],
					   in.stride(), out.stride(),
					   op, nulval);

  // 左下
    int	ys = 1 + 2*threads.y * blocks.y;
    threads.x = threads.y = (out.nrow() - ys - 1) / 2;
    if (threads.x == 0)
	return;
    blocks.x = out.stride() / (2*threads.x);
    blocks.y = 1;
    extrema3x3_kernel<<<blocks, threads>>>((const S*)in[ys], (T*)out[ys],
					   in.stride(), out.stride(),
					   op, nulval);

  // 右下
    if (2*threads.x * blocks.x == out.stride())
	return;
    int	xs = out.stride() - 2*threads.x;
    blocks.x = 1;
    extrema3x3_kernel<<<blocks, threads>>>((const S*)in[ys]  + xs,
					   (	  T*)out[ys] + xs,
					   in.stride(), out.stride(),
					   op, nulval);
}

template  void
cudaSuppressNonExtrema3x3(const CudaArray2<u_char>& in,
			  CudaArray2<u_char>& out,
			  thrust::greater<u_char> op, u_char nulval);
template  void
cudaSuppressNonExtrema3x3(const CudaArray2<float>& in,
			  CudaArray2<float>& out,
			  thrust::greater<float> op, float nulval);
template  void
cudaSuppressNonExtrema3x3(const CudaArray2<u_char>& in,
			  CudaArray2<u_char>& out,
			  thrust::less<u_char> op, u_char nulval);
template  void
cudaSuppressNonExtrema3x3(const CudaArray2<float>& in,
			  CudaArray2<float>& out,
			  thrust::less<float> op, float nulval);
}
