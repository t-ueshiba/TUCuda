/*
 *  $Id$
 */
/*!
  \file		algorithm.h
  \brief	各種アルゴリズムの定義と実装
*/ 
#ifndef TU_CUDA_ALGORITHM_H
#define TU_CUDA_ALGORITHM_H

#include "TU/algorithm.h"
#include "TU/iterator.h"

namespace TU
{
namespace cuda
{
/************************************************************************
*  global constatnt variables						*
************************************************************************/
static constexpr size_t	BlockDimX = 32;	//!< 1ブロックあたりのスレッド数(x方向)
static constexpr size_t	BlockDimY = 16;	//!< 1ブロックあたりのスレッド数(y方向)
static constexpr size_t	BlockDim  = 32;	//!< 1ブロックあたりのスレッド数(全方向)
    
/************************************************************************
*  stride(ROW row)							*
************************************************************************/
//! 与えられた行と次の行の先頭要素間の距離を返す.
/*!
  rowとrowの次の反復子が有効であることが前提条件である.
  \param row	2次元配列の行を指す反復子
  \return	rowとrowの次の行の先頭要素間の距離
*/
template <class ROW> inline ptrdiff_t
stride(ROW row)
{
    auto	next = row;
    return std::distance(row->begin(), (++next)->begin());
}

/************************************************************************
*  copyToConstantMemory(ITER begin, ITER end, T* dst)			*
************************************************************************/
//! CUDAの定数メモリ領域にデータをコピーする．
/*!
  \param begin	コピー元データの先頭を指す反復子
  \param end	コピー元データの末尾の次を指す反復子
  \param dst	コピー先の定数メモリ領域を指すポインタ
*/
template <class ITER, class T> inline void
copyToConstantMemory(ITER begin, ITER end, T* dst)
{
    if (begin < end)
	cudaMemcpyToSymbol(reinterpret_cast<const char*>(dst), &(*begin),
			   std::distance(begin, end)*sizeof(T), 0,
			   cudaMemcpyHostToDevice);
}

/************************************************************************
*  subsample(IN in, IN ie, OUT out)					*
************************************************************************/
//! CUDAによって2次元配列を水平／垂直方向それぞれ1/2に間引く．
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class IN, class OUT> void
subsample(IN in, IN ie, OUT out)					;

#if defined(__NVCC__)
template <class IN, class OUT> static __global__ void
subsample_kernel(IN in, OUT out, int stride_i, int stride_o)
{
    const auto	x = blockIdx.x*blockDim.x + threadIdx.x;
    const auto	y = blockIdx.y*blockDim.y + threadIdx.y;
    
    out[y*stride_o + x] = in[2*(y*stride_i + x)];
}

template <class IN, class OUT> void
subsample(IN in, IN ie, OUT out)
{
    const auto	nrow = std::distance(in, ie)/2;
    if (nrow < 1)
	return;

    const auto	ncol = std::distance(in->begin(), in->end())/2;
    if (ncol < 1)
	return;
	
    const auto	stride_i = stride(in);
    const auto	stride_o = stride(out);
    
  // 左上
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncol/threads.x, nrow/threads.y);
    subsample_kernel<<<blocks, threads>>>(in->begin(), out->begin(),
					  stride_i, stride_o);
  // 右上
    const auto	x = blocks.x*threads.x;
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    subsample_kernel<<<blocks, threads>>>(in->begin() + 2*x, out->begin() + x,
					  stride_i, stride_o);
  // 左下
    std::advance(in, 2*blocks.y*threads.y);
    std::advance(out,  blocks.y*threads.y);
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    threads.y = nrow%threads.y;
    blocks.y  = 1;
    subsample_kernel<<<blocks, threads>>>(in->begin(), out->begin(),
					  stride_i, stride_o);

  // 右下
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    subsample_kernel<<<blocks, threads>>>(in->begin() + 2*x, out->begin() + x,
					  stride_i, stride_o);
}
#endif
    
/************************************************************************
*  op3x3(IN in, IN ie, OUT out, OP op)					*
************************************************************************/
//! CUDAによって2次元配列に対して3x3近傍演算を行う．
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
  \param op	3x3近傍演算子
*/
template <class IN, class OUT, class OP> void
op3x3(IN in, IN ie, OUT out, OP op)					;

#if defined(__NVCC__)
/*
 *  device functions
 */
template <class IN, class OUT, class OP> static __global__ void
op3x3_kernel(IN in, OUT out, OP op, int stride_i, int stride_o)
{
    typedef typename std::iterator_traits<IN>::value_type	value_type;
    
    const int	bx = blockDim.x;
    const int	by = blockDim.y;
    int		xy = (blockIdx.y*by + threadIdx.y)*stride_i 
		   +  blockIdx.x*bx + threadIdx.x;	// 現在位置
    int		x  = 1 + threadIdx.x;
    const int	y  = 1 + threadIdx.y;
    
  // 原画像のブロック内部およびその外枠1画素分を共有メモリに転送
    __shared__ value_type	in_s[BlockDimY+2][BlockDimX+2];
    in_s[y][x] = in[xy];				// 内部

    if (threadIdx.x == 0)	// ブロックの左端?
    {
	in_s[y][0     ] = in[xy -  1];			// 左枠
	in_s[y][1 + bx] = in[xy + bx];			// 右枠
    }
    
    if (threadIdx.y == 0)	// ブロックの上端?
    {
	const int	top = xy - stride_i;		// 現在位置の直上
	const int	bot = xy + by*stride_i;		// 現在位置の下端

	in_s[0     ][x] = in[top];			// 上枠
	in_s[1 + by][x] = in[bot];			// 下枠

	if (threadIdx.x == 0)	// ブロックの左上隅?
	{
	    in_s[0     ][0     ] = in[top -  1];	// 左上隅
	    in_s[0     ][1 + bx] = in[top + bx];	// 右上隅
	    in_s[1 + by][0     ] = in[bot -  1];	// 左下隅
	    in_s[1 + by][1 + bx] = in[bot + bx];	// 右下隅
	}
    }
    __syncthreads();

  // 共有メモリに保存した原画像データから現在画素に対するフィルタ出力を計算
    xy = (blockIdx.y*by + threadIdx.y)*stride_o
       +  blockIdx.x*bx + threadIdx.x;
    --x;
    out[xy] = op(in_s[y - 1] + x, in_s[y] + x, in_s[y + 1] + x);
}

template <class IN, class OUT, class OP> void
op3x3(IN in, IN ie, OUT out, OP op)
{
    const auto	nrow = std::distance(in, ie) - 2;
    if (nrow < 1)
	return;

    const auto	ncol = std::distance(in->begin(), in->end()) - 2;
    if (ncol < 1)
	return;
    
    const auto	stride_i = stride(in);
    const auto	stride_o = stride(out);
    
  // 左上
    ++in;
    ++out;
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncol/threads.x, nrow/threads.y);
    op3x3_kernel<<<blocks, threads>>>(in->begin().get() + 1,
				      out->begin().get() + 1,
				      op, stride_i, stride_o);
  // 右上
    const auto	x = 1 + blocks.x*threads.x;
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    op3x3_kernel<<<blocks, threads>>>(in->begin().get() + x,
				      out->begin().get() + x,
				      op, stride_i, stride_o);
  // 左下
    std::advance(in,  blocks.y*threads.y);
    std::advance(out, blocks.y*threads.y);
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    threads.y = nrow%threads.y;
    blocks.y  = 1;
    op3x3_kernel<<<blocks, threads>>>(in->begin().get() + 1,
				      out->begin().get() + 1,
				      op, stride_i, stride_o);
  // 右下
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    op3x3_kernel<<<blocks, threads>>>(in->begin().get() + x,
				      out->begin().get() + x,
				      op, stride_i, stride_o);
}
#endif
    
/************************************************************************
*  suppressNonExtrema(							*
*      IN in, IN ie, OUT out, OP op,					*
*      typename iterator_traits<IN>::value_type::value_type nulval)	*
************************************************************************/
//! CUDAによって2次元配列に対して3x3非極値抑制処理を行う．
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
  \param op	極大値を検出するときは thrust::greater<T> を，
		極小値を検出するときは thrust::less<T> を与える
  \param nulval	非極値をとる画素に割り当てる値
*/
template <class IN, class OUT, class OP> void
suppressNonExtrema3x3(
    IN in, IN ie, OUT out, OP op,
    typename std::iterator_traits<IN>::value_type::value_type nulval=0)	;

#if defined(__NVCC__)
/*
 *  device functions
 */
template <class IN, class OUT, class OP> static __global__ void
extrema3x3_kernel(IN in, OUT out, OP op,
		  typename std::iterator_traits<IN>::value_type nulval,
		  int stride_i, int stride_o)
{
    typedef typename std::iterator_traits<IN>::value_type	value_type;
    
    const int	bx2 = 2*blockDim.x;
    const int	by2 = 2*blockDim.y;
    int		xy  = 2*((blockIdx.y*blockDim.y + threadIdx.y)*stride_i +
			  blockIdx.x*blockDim.x + threadIdx.x);
    const int	x   = 1 + 2*threadIdx.x;
    const int	y   = 1 + 2*threadIdx.y;

  // 原画像の (2*blockDim.x)x(2*blockDim.y) 矩形領域を共有メモリにコピー
    __shared__ value_type	in_s[2*BlockDimY + 2][2*BlockDimX + 3];
    in_s[y    ][x    ] = in[xy		     ];
    in_s[y    ][x + 1] = in[xy		  + 1];
    in_s[y + 1][x    ] = in[xy + stride_i    ];
    in_s[y + 1][x + 1] = in[xy + stride_i + 1];

  // 2x2ブロックの外枠を共有メモリ領域にコピー
    if (threadIdx.x == 0)	// ブロックの左端?
    {
	const int	lft = xy - 1;
	const int	rgt = xy + bx2;
	
	in_s[y    ][0      ] = in[lft		];	// 左枠上
	in_s[y + 1][0      ] = in[lft + stride_i];	// 左枠下
	in_s[y    ][1 + bx2] = in[rgt		];	// 右枠上
	in_s[y + 1][1 + bx2] = in[rgt + stride_i];	// 右枠下半
    }
    
    if (threadIdx.y == 0)	// ブロックの上端?
    {
	const int	top  = xy - stride_i;		// 現在位置の直上
	const int	bot  = xy + by2*stride_i;	// 現在位置の下端

	in_s[0	    ][x    ] = in[top    ];		// 上枠左
	in_s[0	    ][x + 1] = in[top + 1];		// 上枠右
	in_s[1 + by2][x    ] = in[bot    ];		// 下枠左
	in_s[1 + by2][x + 1] = in[bot + 1];		// 下枠右

	if (threadIdx.x == 0)	// ブロックの左上隅?
	{
	    in_s[0      ][0      ] = in[top -   1];	// 左上隅
	    in_s[0      ][1 + bx2] = in[top + bx2];	// 右上隅
	    in_s[1 + by2][0      ] = in[bot -   1];	// 左下隅
	    in_s[1 + by2][1 + bx2] = in[bot + bx2];	// 右下隅
	}
    }
    __syncthreads();

  // このスレッドの処理対象である2x2ウィンドウ中で最大/最小となる画素の座標を求める．
  //const int	i01 = (op(in_s[y    ][x], in_s[y    ][x + 1]) ? 0 : 1);
  //const int	i23 = (op(in_s[y + 1][x], in_s[y + 1][x + 1]) ? 2 : 3);
    const int	i01 = op(in_s[y    ][x + 1], in_s[y    ][x]);
    const int	i23 = op(in_s[y + 1][x + 1], in_s[y + 1][x]) + 2;
    const int	iex = (op(in_s[y][x + i01], in_s[y + 1][x + (i23 & 0x1)]) ?
		       i01 : i23);
    const int	xx  = x + (iex & 0x1);		// 最大/最小点のx座標
    const int	yy  = y + (iex >> 1);		// 最大/最小点のy座標

  // 最大/最小となった画素が，残り5つの近傍点よりも大きい/小さいか調べる．
  //const int	dx  = (iex & 0x1 ? 1 : -1);
  //const int	dy  = (iex & 0x2 ? 1 : -1);
    const int	dx  = ((iex & 0x1) << 1) - 1;
    const int	dy  = (iex & 0x2) - 1;
    auto	val = in_s[yy][xx];
    val = (op(val, in_s[yy + dy][xx - dx]) &
	   op(val, in_s[yy + dy][xx     ]) &
	   op(val, in_s[yy + dy][xx + dx]) &
	   op(val, in_s[yy     ][xx + dx]) &
	   op(val, in_s[yy - dy][xx + dx]) ? val : nulval);
    __syncthreads();

  // この2x2画素ウィンドウに対応する共有メモリ領域に出力値を書き込む．
    in_s[y    ][x    ] = nulval;		// 非極値
    in_s[y    ][x + 1] = nulval;		// 非極値
    in_s[y + 1][x    ] = nulval;		// 非極値
    in_s[y + 1][x + 1] = nulval;		// 非極値
    in_s[yy   ][xx   ] = val;			// 極値または非極値
    __syncthreads();

  // (2*blockDim.x)x(2*blockDim.y) の矩形領域に共有メモリ領域をコピー．
    xy  = 2*((blockIdx.y*blockDim.y + threadIdx.y)*stride_o +
	      blockIdx.x*blockDim.x + threadIdx.x);
    out[xy		 ] = in_s[y    ][x    ];
    out[xy	      + 1] = in_s[y    ][x + 1];
    out[xy + stride_o	 ] = in_s[y + 1][x    ];
    out[xy + stride_o + 1] = in_s[y + 1][x + 1];
}
    
template <class IN, class OUT, class OP> void
suppressNonExtrema3x3(
    IN in, IN ie, OUT out, OP op,
    typename std::iterator_traits<IN>::value_type::value_type nulval)
{
    const auto	nrow = (std::distance(in, ie) - 1)/2;
    if (nrow < 1)
	return;

    const auto	ncol = (std::distance(in->begin(), in->end()) - 1)/2;
    if (ncol < 1)
	return;
    
    const auto	stride_i = stride(in);
    const auto	stride_o = stride(out);
    
  // 左上
    ++in;
    ++out;
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncol/threads.x, nrow/threads.y);
    extrema3x3_kernel<<<blocks, threads>>>(in->begin(), out->begin(),
					   op, nulval, stride_i, stride_o);
  // 右上
    const auto	x = blocks.x*threads.x;
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    extrema3x3_kernel<<<blocks, threads>>>(in->begin() + x, out->begin() + x,
					   op, nulval, stride_i, stride_o);
  // 左下
    std::advance(in,  blocks.y*(2*threads.y));
    std::advance(out, blocks.y*(2*threads.y));
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    threads.y = nrow%threads.y;
    blocks.y  = 1;
    extrema3x3_kernel<<<blocks, threads>>>(in->begin(), out->begin(),
					   op, nulval, stride_i, stride_o);
  // 右下
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    extrema3x3_kernel<<<blocks, threads>>>(in->begin() + x, out->begin() + x,
					   op, nulval, stride_i, stride_o);
}
#endif

/************************************************************************
*  transpose(IN in, IN ie, OUT out)					*
************************************************************************/
//! CUDAによって2次元配列の転置処理を行う．
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class IN, class OUT> void
transpose(IN in, IN ie, OUT out)					;
    
#if defined(__NVCC__)
namespace detail
{
template <class IN, class OUT> static __global__ void
transpose_kernel(IN in, OUT out, int stride_i, int stride_o)
{
    typedef typename std::iterator_traits<IN>::value_type	value_type;

    const auto			bx = blockIdx.x*blockDim.x;
    const auto			by = blockIdx.y*blockDim.y;
    __shared__ value_type	tile[BlockDim][BlockDim + 1];
    tile[threadIdx.y][threadIdx.x]
	= in[(by + threadIdx.y)*stride_i + bx + threadIdx.x];
    __syncthreads();
    out[(bx + threadIdx.y)*stride_o + by + threadIdx.x]
	= tile[threadIdx.x][threadIdx.y];
}
    
template <class IN, class OUT> static void
transpose(IN in, IN ie, OUT out, size_t i, size_t j)
{
    size_t	r = std::distance(in, ie);
    if (r < 1)
	return;

    size_t	c = std::distance(std::cbegin(*in), std::cend(*in)) - j;
    if (c < 1)
	return;

    const auto	stride_i = stride(in);
    const auto	stride_o = stride(out);
    const auto	blockDim = std::min(BlockDim, r, c);
    const dim3	threads(blockDim, blockDim);
    const dim3	blocks(c/threads.x, r/threads.y);
    transpose_kernel<<<blocks, threads>>>(std::cbegin(*in) + j,
					  std::begin(*out) + i,
					  stride_i, stride_o);	// 左上

    r = blocks.y*threads.y;
    c = blocks.x*threads.x;

    auto	in_n = in;
    std::advance(in_n, r);
    auto	out_n = out;
    std::advance(out_n, c);
    transpose(in,   in_n, out_n, i,     j + c);			// 右上
    transpose(in_n, ie,   out,   i + r, j);			// 下
}

}	// namesapce detail
    
template <class IN, class OUT> inline void
transpose(IN in, IN ie, OUT out)
{
    detail::transpose(in, ie, out, 0, 0);
}
#endif    

}	// namespace cuda
}	// namespace TU
#endif	// !__CUDA_ALGORITHM_H

