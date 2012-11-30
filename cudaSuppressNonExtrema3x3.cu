/*
 *  $Id: cudaSuppressNonExtrema3x3.cu,v 1.2 2011-05-09 00:35:49 ueshiba Exp $
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
template <class T, class OP> static __global__ void
extrema3x3_kernel(const T* in, T* out,
		  u_int stride_i, u_int stride_o, OP op, T nulval)
{
  // このカーネルはブロック境界処理のために blockDim.x == blockDim.y を仮定
    const int	blk = blockDim.x;	// u_intにするとダメ．CUDAのバグ？
    int		xy  = 2*(blockIdx.y*blockDim.y + threadIdx.y)*stride_i
		    + 2* blockIdx.x*blockDim.x + threadIdx.x;
    int		x   = 1 +   threadIdx.x;
    const int	y   = 1 + 2*threadIdx.y;

  // (2*blockDim.x)x(2*blockDim.y) の矩形領域に共有メモリ領域にコピー
    __shared__ T	buf[2*BlockDim + 2][2*BlockDim + 3];
    buf[y    ][x      ] = in[xy			];
    buf[y    ][x + blk] = in[xy		   + blk];
    buf[y + 1][x      ] = in[xy + stride_i	];
    buf[y + 1][x + blk] = in[xy + stride_i + blk];

  // 2x2ブロックの外枠を共有メモリ領域にコピー
    if (threadIdx.y == 0)	// ブロックの上端?
    {
	const int	blk2 = 2*blockDim.y;
	const int	top  = xy - stride_i;		// 現在位置の直上
	const int	bot  = xy + blk2 * stride_i;	// 現在位置の下端
	buf[0	    ][x      ] = in[top      ];		// 上枠左半
	buf[0	    ][x + blk] = in[top + blk];		// 上枠右半
	buf[1 + blk2][x      ] = in[bot      ];		// 下枠左半
	buf[1 + blk2][x + blk] = in[bot + blk];		// 下枠右半

	int	lft = xy + threadIdx.x*(stride_i - 1);
	buf[x      ][0       ] = in[lft -    1];	// 左枠上半
	buf[x      ][1 + blk2] = in[lft + blk2];	// 右枠上半
	lft += blockDim.y * stride_i;
	buf[x + blk][0       ] = in[lft -    1];	// 左枠下半
	buf[x + blk][1 + blk2] = in[lft + blk2];	// 右枠下半

	if (threadIdx.x == 0)	// ブロックの左上隅?
	{
	    if ((blockIdx.x != 0) || (blockIdx.y != 0))
	  	buf[0][0] = in[top - 1];			// 左上隅
	    if ((blockIdx.x != gridDim.x - 1) || (blockIdx.y != gridDim.y - 1))
		buf[1 + blk2][1 + blk2] = in[bot + blk2];	// 右下隅
	    buf[0][1 + blk2] = in[top + blk2];			// 右上隅
	    buf[1 + blk2][0] = in[bot -    1];			// 左下隅
	}
    }
    __syncthreads();

  // このスレッドの処理対象である2x2ウィンドウ中で最大/最小となる画素の座標を求める．
    x = 1 + 2*threadIdx.x;
  //const int	i01 = (op(buf[y    ][x], buf[y	  ][x + 1]) ? 0 : 1);
  //const int	i23 = (op(buf[y + 1][x], buf[y + 1][x + 1]) ? 2 : 3);
    const int	i01 = op(buf[y    ][x + 1], buf[y    ][x]);
    const int	i23 = op(buf[y + 1][x + 1], buf[y + 1][x]) + 2;
    const int	iex = (op(buf[y][x + i01], buf[y + 1][x + (i23 & 0x1)]) ?
		       i01 : i23);
    const int	xx  = x + (iex & 0x1);		// 最大/最小点のx座標
    const int	yy  = y + (iex >> 1);		// 最大/最小点のy座標

  // 最大/最小となった画素が，残り5つの近傍点よりも大きい/小さいか調べる．
  //const int	dx  = (iex & 0x1 ? 1 : -1);
  //const int	dy  = (iex & 0x2 ? 1 : -1);
    const int	dx  = ((iex & 0x1) << 1) - 1;
    const int	dy  = (iex & 0x2) - 1;
    T		val = buf[yy][xx];
    val = (op(val, buf[yy + dy][xx - dx]) &
	   op(val, buf[yy + dy][xx     ]) &
	   op(val, buf[yy + dy][xx + dx]) &
	   op(val, buf[yy     ][xx + dx]) &
	   op(val, buf[yy - dy][xx + dx]) ? val : nulval);
    __syncthreads();

  // この2x2画素ウィンドウに対応する共有メモリ領域に出力値を書き込む．
    buf[y    ][x    ] = nulval;		// 非極値
    buf[y    ][x + 1] = nulval;		// 非極値
    buf[y + 1][x    ] = nulval;		// 非極値
    buf[y + 1][x + 1] = nulval;		// 非極値
    buf[yy   ][xx   ] = val;		// 極値または非極値
    __syncthreads();
    
  // (2*blockDim.x)x(2*blockDim.y) の矩形領域に共有メモリ領域をコピー．
    x = 1 + threadIdx.x;
    xy  = 2*(blockIdx.y*blockDim.y + threadIdx.y)*stride_o
	+ 2* blockIdx.x*blockDim.x + threadIdx.x;
    out[xy		   ] = buf[y    ][x      ];
    out[xy	      + blk] = buf[y    ][x + blk];
    out[xy + stride_o	   ] = buf[y + 1][x      ];
    out[xy + stride_o + blk] = buf[y + 1][x + blk];
}
    
/************************************************************************
*  global functions							*
************************************************************************/
//! CUDAによって2次元配列に対して3x3非極値抑制処理を行う．
/*!
  \param in	入力2次元配列
  \param out	出力2次元配列
  \param op	極大値を検出するときは thrust::greater<T> を，
		極小値を検出するときは thrust::less<T> を与える
  \param nulval	非極値をとる画素に割り当てる値
*/
template <class T, class OP> void
cudaSuppressNonExtrema3x3(const CudaArray2<T>& in,
				CudaArray2<T>& out, OP op, T nulval)
{
    if (in.nrow() < 3 || in.ncol() < 3)
	return;
    
    out.resize(in.nrow(), in.ncol());

  // 最初と最後の行を除いた (out.nrow() - 2) x out.stride() の配列として扱う
    dim3	threads(BlockDim, BlockDim);
    dim3	blocks(out.stride()     / (2*threads.x),
		       (out.nrow() - 2) / (2*threads.y));
    extrema3x3_kernel<<<blocks, threads>>>(in[1].ptr(), out[1].ptr(),
					   in.stride(), out.stride(),
					   op, nulval);

  // 左下
    int	ys = 1 + 2*threads.y * blocks.y;
    threads.x = threads.y = (out.nrow() - ys - 1) / 2;
    if (threads.x == 0)
	return;
    blocks.x = out.stride() / (2*threads.x);
    blocks.y = 1;
    extrema3x3_kernel<<<blocks, threads>>>(in[ys].ptr(), out[ys].ptr(),
					   in.stride(), out.stride(),
					   op, nulval);

  // 右下
    if (2*threads.x * blocks.x == out.stride())
	return;
    int	xs = out.stride() - 2*threads.x;
    blocks.x = 1;
    extrema3x3_kernel<<<blocks, threads>>>(in[ys].ptr()  + xs,
					   out[ys].ptr() + xs,
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
