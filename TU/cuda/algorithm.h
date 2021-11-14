/*
 *  $Id$
 */
/*!
  \file		algorithm.h
  \brief	各種アルゴリズムの定義と実装
*/
#pragma once

#include "TU/cuda/allocator.h"
#include "TU/cuda/iterator.h"
#include "TU/cuda/vec.h"

namespace TU
{
namespace cuda
{
#if defined(__NVCC__)
//! デバイス関数を納める名前空間
namespace device
{
  /**********************************************************************
  *  __device__ functions						*
  **********************************************************************/
  //! スレッドブロック中のラインに指定された長さを付加した1次元領域をコピーする
  /*!
    \param src		コピー元のラインの左端を指す反復子
    \param dst		コピー先の1次元配列
    \param dx		コピー元のライン幅に付加される長さ
  */
  template <class S, class T> __device__ static inline void
  loadLine(S src, T dst[], int dx)
  {
      auto	tx = threadIdx.x;
      dx += blockDim.x;
      do
      {
	  dst[tx] = src[tx];
      } while ((tx += blockDim.x) < dx);
  }

  //! スレッドブロックの横方向に指定された長さを付加した領域をコピーする
  /*!
    \param src		コピー元の矩形領域の左上隅を指す反復子
    \param stride	コピー元の行を1つ進めるためのインクリメント数
    \param dst		コピー先の2次元配列
    \param dx		ブロック幅に付加される長さ
  */
  template <class S, class STRIDE, class T, size_t W>
  __device__ static inline void
  loadTileH(S src, STRIDE stride, T dst[][W], int dx)
  {
      advance_stride(src, threadIdx.y * stride);

      auto		tx = threadIdx.x;
      const auto	q  = dst[threadIdx.y];
      dx += blockDim.x;
      do
      {
	  q[tx] = src[tx];
      } while ((tx += blockDim.x) < dx);
  }

  //! スレッドブロックの縦方向に指定された長さを付加した領域をコピーする
  /*!
    \param src		コピー元の矩形領域の左上隅を指す反復子
    \param stride	コピー元の行を1つ進めるためのインクリメント数
    \param dst		コピー先の2次元配列
    \param dy		ブロック高に付加される長さ
  */
  template <class S, class STRIDE, class T, size_t W>
  __device__ static inline void
  loadTileV(S src, STRIDE stride, T dst[][W], int dy)
  {
      auto		ty = threadIdx.y;
      advance_stride(src, ty * stride);
      src += threadIdx.x;

      dy += blockDim.y;
      do
      {
	  dst[ty][threadIdx.x] = *src;
	  advance_stride(src, blockDim.y * stride);
      } while ((ty += blockDim.y) < dy);
  }

  //! スレッドブロックの縦方向に指定された長さを付加した領域を転置してコピーする
  /*!
    コピー先の矩形領域のサイズは blockDim.x * (blockDim.y + dy) となる．
    \param src		コピー元の矩形領域の左上隅を指す反復子
    \param stride	コピー元の行を1つ進めるためのインクリメント数
    \param dst		コピー先の2次元配列
    \param dy		ブロック高に付加される長さ
  */
  template <class S, class STRIDE, class T, size_t W>
  __device__ static inline void
  loadTileVt(S src, STRIDE stride, T dst[][W], int dy)
  {
      auto		ty = threadIdx.y;
      advance_stride(src, ty * stride);
      src += threadIdx.x;

      const auto	q = dst[threadIdx.x];
      dy += blockDim.y;
      do
      {
	  q[ty] = *src;
	  advance_stride(src, blockDim.y * stride);
      } while ((ty += blockDim.y) < dy);
  }

  //! スレッドブロックの横方向と縦方向にそれぞれ指定された長さを付加した領域をコピーする
  /*!
    \param src		コピー元の矩形領域の左上隅を指す反復子
    \param stride	コピー元の行を1つ進めるためのインクリメント数
    \param dst		コピー先の2次元配列
    \param dx		ブロック幅に付加される長さ
    \param dy		ブロック高に付加される長さ
  */
  template <class S, class STRIDE, class T, size_t W>
  __device__ static inline void
  loadTile(S src, STRIDE stride, T dst[][W], int dx, int dy)
  {
      auto	ty = threadIdx.y;
      advance_stride(src, ty * stride);

      dx += blockDim.x;
      dy += blockDim.y;
      do
      {
	  auto	tx = threadIdx.x;
	  do
	  {
	      dst[ty][tx] = src[tx];
	  } while ((tx += blockDim.x) < dx);
	  advance_stride(src, blockDim.y * stride);
      } while ((ty += blockDim.y) < dy);
  }
}	// namespace device
#endif

/************************************************************************
*  global constatnt variables						*
************************************************************************/
constexpr static size_t	BlockDimX = 32;	//!< 1ブロックあたりのスレッド数(x方向)
constexpr static size_t	BlockDimY = 16;	//!< 1ブロックあたりのスレッド数(y方向)
constexpr static size_t	BlockDim  = 32;	//!< 1ブロックあたりのスレッド数(全方向)
constexpr static size_t	BlockDim1 = BlockDimX * BlockDimX;

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
namespace device
{
  template <class IN, class OUT, class STRIDE_I, class STRIDE_O>
  __global__ static void
  subsample(IN in, OUT out, STRIDE_I stride_i, STRIDE_O stride_o)
  {
      const int	x = blockIdx.x*blockDim.x + threadIdx.x;
      const int	y = blockIdx.y*blockDim.y + threadIdx.y;

      advance_stride(in, 2*y*stride_i);
      advance_stride(out,  y*stride_o);
      out[x] = in[2*x];
  }
}	// namespace device

template <class IN, class OUT> void
subsample(IN in, IN ie, OUT out)
{
    using	std::cbegin;
    using	std::cend;
    using	std::begin;

    const auto	nrow = std::distance(in, ie)/2;
    if (nrow < 1)
	return;

    const auto	ncol = std::distance(cbegin(*in), cend(*in))/2;
    if (ncol < 1)
	return;

    const auto	stride_i = stride(in);
    const auto	stride_o = stride(out);

  // 左上
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncol/threads.x, nrow/threads.y);
    device::subsample<<<blocks, threads>>>(
	get(cbegin(*in)), get(begin(*out)), stride_i, stride_o);
  // 右上
    const auto	x = blocks.x*threads.x;
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::subsample<<<blocks, threads>>>(
	get(cbegin(*in) + 2*x), get(begin(*out) + x), stride_i, stride_o);
  // 左下
    std::advance(in, 2*blocks.y*threads.y);
    std::advance(out,  blocks.y*threads.y);
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    threads.y = nrow%threads.y;
    blocks.y  = 1;
    device::subsample<<<blocks, threads>>>(
	get(cbegin(*in)), get(begin(*out)), stride_i, stride_o);
  // 右下
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::subsample<<<blocks, threads>>>(
	get(cbegin(*in) + 2*x), get(begin(*out) + x), stride_i, stride_o);
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
namespace device
{
  template <class IN, class OUT, class OP, class STRIDE_I, class STRIDE_O>
  __global__ static void
  op3x3(IN in, OUT out, OP op, STRIDE_I stride_i, STRIDE_O stride_o)
  {
      using	value_type = typename std::iterator_traits<IN>::value_type;

      const auto	bx  = blockDim.x;
      const auto	by  = blockDim.y;
      const auto	x0 = __mul24(blockIdx.x, bx);
      const auto	y0 = __mul24(blockIdx.y, by);
      const auto	tx  = threadIdx.x;
      const auto	ty  = threadIdx.y;

      advance_stride(in, y0 * stride_i);
      advance_stride(out, (y0 + ty + 1)*stride_o);

    // 原画像のブロック内部およびその外枠1画素分を共有メモリに転送
      __shared__ value_type	in_s[BlockDimY+2][BlockDimX+2];
      loadTile(in + x0, stride_i, in_s, 2, 2);
      __syncthreads();

    // 共有メモリに保存した原画像データから現在画素に対するフィルタ出力を計算
      out[x0 + tx + 1]
	  = op(in_s[ty] + tx, in_s[ty + 1] + tx, in_s[ty + 2] + tx);
  }
}	// namespace device

template <class IN, class OUT, class OP> void
op3x3(IN in, IN ie, OUT out, OP op)
{
    using	std::cbegin;
    using	std::cend;
    using	std::begin;

    const auto	nrow = std::distance(in, ie) - 2;
    if (nrow < 1)
	return;

    const auto	ncol = std::distance(cbegin(*in), cend(*in)) - 2;
    if (ncol < 1)
	return;

    const auto	stride_i = stride(in);
    const auto	stride_o = stride(out);

  // 左上
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncol/threads.x, nrow/threads.y);
    device::op3x3<<<blocks, threads>>>(get(cbegin(*in)),
				       get(begin(*out)),
				       op, stride_i, stride_o);
  // 右上
    const auto	x = blocks.x*threads.x;
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::op3x3<<<blocks, threads>>>(get(cbegin(*in) + x),
				       get(begin(*out) + x),
				       op, stride_i, stride_o);
  // 左下
    std::advance(in,  blocks.y*threads.y);
    std::advance(out, blocks.y*threads.y);
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    threads.y = nrow%threads.y;
    blocks.y  = 1;
    device::op3x3<<<blocks, threads>>>(get(cbegin(*in)),
				       get(begin(*out)),
				       op, stride_i, stride_o);
  // 右下
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::op3x3<<<blocks, threads>>>(get(cbegin(*in) + x),
				       get(begin(*out) + x),
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
namespace device
{
  template <class IN, class OUT, class OP> __global__ static void
  extrema3x3(IN in, OUT out, OP op,
		    typename std::iterator_traits<IN>::value_type nulval,
		    int stride_i, int stride_o)
  {
      using	value_type = typename std::iterator_traits<IN>::value_type;

    // in[] の index は負になり得るので，index 計算に
    // 使われる xy 等の変数の型は符号付きでなければならない．
      const int	bx2 = 2*blockDim.x;
      const int	by2 = 2*blockDim.y;
      int	xy  = 2*((blockIdx.y*blockDim.y + threadIdx.y)*stride_i +
			 blockIdx.x*blockDim.x + threadIdx.x);
      const int	x   = 1 + 2*threadIdx.x;
      const int	y   = 1 + 2*threadIdx.y;

  // 原画像の (2*blockDim.x)x(2*blockDim.y) 矩形領域を共有メモリにコピー
      __shared__ value_type	in_s[2*BlockDimY + 2][2*BlockDimX + 3];
      in_s[y    ][x    ] = in[xy	       ];
      in_s[y    ][x + 1] = in[xy	    + 1];
      in_s[y + 1][x    ] = in[xy + stride_i    ];
      in_s[y + 1][x + 1] = in[xy + stride_i + 1];

    // 2x2ブロックの外枠を共有メモリ領域にコピー
      if (threadIdx.x == 0)	// ブロックの左端?
      {
	  const int	lft = xy - 1;
	  const int	rgt = xy + bx2;

	  in_s[y    ][0      ] = in[lft		  ];	// 左枠上
	  in_s[y + 1][0      ] = in[lft + stride_i];	// 左枠下
	  in_s[y    ][1 + bx2] = in[rgt		  ];	// 右枠上
	  in_s[y + 1][1 + bx2] = in[rgt + stride_i];	// 右枠下半
      }

      if (threadIdx.y == 0)	// ブロックの上端?
      {
	  const int	top  = xy - stride_i;		// 現在位置の直上
	  const int	bot  = xy + by2*stride_i;	// 現在位置の下端

	  in_s[0      ][x    ] = in[top    ];	// 上枠左
	  in_s[0      ][x + 1] = in[top + 1];	// 上枠右
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
      const int	xx  = x + (iex & 0x1);			// 最大/最小点のx座標
      const int	yy  = y + (iex >> 1);			// 最大/最小点のy座標

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
      out[xy		   ] = in_s[y    ][x    ];
      out[xy	        + 1] = in_s[y    ][x + 1];
      out[xy + stride_o	   ] = in_s[y + 1][x    ];
      out[xy + stride_o + 1] = in_s[y + 1][x + 1];
  }
}	// namespace device

template <class IN, class OUT, class OP> void
suppressNonExtrema3x3(
    IN in, IN ie, OUT out, OP op,
    typename std::iterator_traits<IN>::value_type::value_type nulval)
{
    using	std::cbegin;
    using	std::cend;
    using	std::begin;

    const auto	nrow = (std::distance(in, ie) - 1)/2;
    if (nrow < 1)
	return;

    const auto	ncol = (std::distance(cbegin(*in), cend(*in)) - 1)/2;
    if (ncol < 1)
	return;

    const auto	stride_i = stride(in);
    const auto	stride_o = stride(out);

  // 左上
    ++in;
    ++out;
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncol/threads.x, nrow/threads.y);
    device::extrema3x3<<<blocks, threads>>>(get(cbegin(*in)),
					    get(begin(*out)),
					    op, nulval, stride_i, stride_o);
  // 右上
    const int	x = blocks.x*threads.x;
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::extrema3x3<<<blocks, threads>>>(get(cbegin(*in) + x),
					    get(begin(*out) + x),
					    op, nulval, stride_i, stride_o);
  // 左下
    std::advance(in,  blocks.y*(2*threads.y));
    std::advance(out, blocks.y*(2*threads.y));
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    threads.y = nrow%threads.y;
    blocks.y  = 1;
    device::extrema3x3<<<blocks, threads>>>(get(cbegin(*in)),
					    get(begin(*out)),
					    op, nulval, stride_i, stride_o);
  // 右下
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::extrema3x3<<<blocks, threads>>>(get(cbegin(*in) + x),
					    get(begin(*out) + x),
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
namespace device
{
  template <class IN, class OUT, class STRIDE_I, class STRIDE_O>
  __global__ static void
  transpose(IN in, OUT out, STRIDE_I stride_i, STRIDE_O stride_o)
  {
      using	value_type = typename std::iterator_traits<IN>::value_type;

      const auto		bx = blockIdx.x*blockDim.x;
      const auto		by = blockIdx.y*blockDim.y;
      __shared__ value_type	tile[BlockDim][BlockDim + 1];

      advance_stride(in, (by + threadIdx.y)*stride_i);
      tile[threadIdx.y][threadIdx.x] = in[bx + threadIdx.x];
      __syncthreads();

      advance_stride(out, (bx + threadIdx.y)*stride_o);
      out[by + threadIdx.x] = tile[threadIdx.x][threadIdx.y];
  }
}	// namespace device

namespace detail
{
  template <class IN, class OUT> static void
  transpose(IN in, IN ie, OUT out, size_t i, size_t j)
  {
      using	std::cbegin;
      using	std::cend;
      using	std::begin;

      size_t	r = std::distance(in, ie);
      if (r < 1)
	  return;

      size_t	c = std::distance(cbegin(*in), cend(*in)) - j;
      if (c < 1)
	  return;

      const auto	stride_i = stride(in);
      const auto	stride_o = stride(out);
      const auto	blockDim = std::min({BlockDim, r, c});
      const dim3	threads(blockDim, blockDim);
      const dim3	blocks(c/threads.x, r/threads.y);
      cuda::device::transpose<<<blocks, threads>>>(get(cbegin(*in) + j),
						   get(begin(*out) + i),
						   stride_i, stride_o); // 左上

      r = blocks.y*threads.y;
      c = blocks.x*threads.x;

      auto	in_n = in;
      std::advance(in_n, r);
      auto	out_n = out;
      std::advance(out_n, c);
      transpose(in,   in_n, out_n, i,     j + c);		// 右上
      transpose(in_n, ie,   out,   i + r, j);			// 下
  }
}	// namesapce detail

template <class IN, class OUT> inline void
transpose(IN in, IN ie, OUT out)
{
    detail::transpose(in, ie, out, 0, 0);
}
#endif

/************************************************************************
*  canonical_xy(OUT out, OUT oe), depth_to_xyz(IN in, IN ie, OUT out)	*
************************************************************************/
template <class ITER_K, class ITER_D> void
set_intrinsic_parameters(ITER_K K, ITER_D d)				;

template <class OUT> void
canonical_xy(OUT out, OUT oe)						;

template <class IN, class OUT> void
depth_to_xyz(IN in, IN ie, OUT out)					;

#if defined(__NVCC__)
namespace device
{
  /*
   *  Static __constant__ variables
   */
  template <class T> __constant__ static vec<T, 2>	_flen[1];
  template <class T> __constant__ static vec<T, 2>	_uv0[1];
  template <class T> __constant__ static T		_d[4];

  /*
   *  Static __device__ functions
   */
  template <class T>
  struct undistort
  {
      __device__ vec<T, 2>
      operator ()(const vec<T, 2>& uv) const
      {
	  static constexpr T	MAX_ERR  = 0.001*0.001;
	  static constexpr int	MAX_ITER = 5;

	  auto		xy  = (uv - _uv0<T>[0])/_flen<T>[0];
	  const auto	xy0 = xy;

	// compensate distortion iteratively
	  for (int n = 0; n < MAX_ITER; ++n)
	  {
	      const auto	r2 = cuda::square(xy);
	      const auto	k  = T(1) + (_d<T>[0] + _d<T>[1]*r2)*r2;
	      if (k < T(0))
		  break;

	      const auto a = T(2) * xy.x * xy.y;
	      vec<T, 2>	 delta{_d<T>[2]*a + _d<T>[3]*(r2 + T(2)*xy.x*xy.x),
			       _d<T>[2]*(r2 + T(2)*xy.y*xy.y) + _d<T>[3]*a};
	      const auto uv_proj = _flen<T>[0]*(k*xy + delta) + _uv0<T>[0];

	      if (cuda::square(uv_proj - uv) < MAX_ERR)
		  break;

	      xy = (xy0 - delta)/k;	// compensate lens distortion
	  }

	  return xy;
      }

      __device__ vec<T, 3>
      operator ()(const vec<T, 3>& uvd) const
      {
	  const auto	xy = (*this)(vec<T, 2>{uvd.x, uvd.y});
	  return {uvd.z*xy.x, uvd.z*xy.y, uvd.z};
      }
  };

  /*
   *  Static __global__ functions
   */
  template <class OUT, class STRIDE> __global__ static void
  canonical_xy(OUT xy, STRIDE stride)
  {
      using element_type = element_t<typename std::iterator_traits<OUT>
						 ::value_type>;

      const int	u = blockIdx.x*blockDim.x + threadIdx.x;
      const int	v = blockIdx.y*blockDim.y + threadIdx.y;

      advance_stride(xy, v*stride);

      xy[u] = undistort<element_type>()({element_type(u), element_type(v)});
  }

  template <class IN, class OUT, class STRIDE_I, class STRIDE_O>
  __global__ static void
  depth_to_points(IN depth, OUT point,
		  STRIDE_I stride_i, STRIDE_O stride_o)
  {
      using element_type = typename std::iterator_traits<IN>::value_type;
      
      const int	u = blockIdx.x*blockDim.x + threadIdx.x;
      const int	v = blockIdx.y*blockDim.y + threadIdx.y;

      advance_stride(depth, v*stride_i);
      advance_stride(point, v*stride_o);

      point[u] = undistort<element_type>()(vec<element_type, 3>{element_type(u),
					    element_type(v), depth[u]});
  }
}	// namespace device

template <class T, class ITER_K, class ITER_D> void
set_intrinsic_parameters(ITER_K K, ITER_D d)
{
    vec<T, 2>	flen, uv0;
    flen.x = *K;
    std::advance(K, 2);
    uv0.x = *K;
    std::advance(K, 2);
    flen.y = *K;
    ++K;
    uv0.y = *K;
    const T	dd[] = {*d, *++d, *++d, *++d};

    cudaMemcpyToSymbol(device::_flen<T>, &flen, sizeof(flen), 0,
    		       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(device::_uv0<T>, &uv0, sizeof(uv0), 0,
		       cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(device::_d<T>,  dd, sizeof(dd), 0,
		       cudaMemcpyHostToDevice);
}

template <class OUT> void
canonical_xy(OUT xy, OUT xy_e)
{
    using	std::begin;
    using	std::end;

    const auto	nrow = std::distance(xy, xy_e);
    if (nrow < 1)
	return;

    const auto	ncol = std::distance(begin(*xy), end(*xy));
    if (ncol < 1)
	return;

    const auto	stride_o = stride(xy);

  // 左上
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncol/threads.x, nrow/threads.y);
    device::canonical_xy<<<blocks, threads>>>(get(begin(*xy)), stride_o);

  // 右上
    const auto	x = blocks.x*threads.x;
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::canonical_xy<<<blocks, threads>>>(get(begin(*xy) + x), stride_o);

  // 左下
    std::advance(xy, blocks.y*threads.y);
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    threads.y = nrow%threads.y;
    blocks.y  = 1;
    device::canonical_xy<<<blocks, threads>>>(get(begin(*xy)), stride_o);

  // 右下
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::canonical_xy<<<blocks, threads>>>(get(begin(*xy) + x), stride_o);
}

template <class IN, class OUT> void
depth_to_points(IN depth, IN depth_e, OUT point)
{
    using	std::cbegin;
    using	std::cend;
    using	std::begin;

    const auto	nrow = std::distance(depth, depth_e);
    if (nrow < 1)
	return;

    const auto	ncol = std::distance(cbegin(*depth), cend(*depth));
    if (ncol < 1)
	return;

    const auto	stride_i = stride(depth);
    const auto	stride_o = stride(point);

  // 左上
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncol/threads.x, nrow/threads.y);
    device::depth_to_points<<<blocks, threads>>>(get(cbegin(*depth)),
						 get( begin(*point)),
						 stride_i, stride_o);
  // 右上
    const auto	x = blocks.x*threads.x;
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::depth_to_points<<<blocks, threads>>>(get(cbegin(*depth) + x),
						 get( begin(*point) + x),
						 stride_i, stride_o);
  // 左下
    std::advance(depth, blocks.y*threads.y);
    std::advance(point, blocks.y*threads.y);
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    threads.y = nrow%threads.y;
    blocks.y  = 1;
    device::depth_to_points<<<blocks, threads>>>(get(cbegin(*depth)),
						 get( begin(*point)),
						 stride_i, stride_o);
  // 右下
    threads.x = ncol%threads.x;
    blocks.x  = 1;
    device::depth_to_points<<<blocks, threads>>>(get(cbegin(*depth) + x),
						 get( begin(*point) + x),
						 stride_i, stride_o);
}
#endif

/************************************************************************
*  eigenvalue/eigenvector of 3x3 matrices				*
************************************************************************/
#if defined(__NVCC__)
namespace device
{
  template <class T> __device__ inline vec<T, 3>
  cardano(const mat3x<T, 3>& A)
  {
    // Determine coefficients of characteristic poynomial. We write
    //       | a   d   f  |
    //  A =  | d*  b   e  |
    //       | f*  e*  c  |
      const T de = A.x.y * A.y.z;	    // d * e
      const T dd = square(A.x.y);	    // d^2
      const T ee = square(A.y.z);	    // e^2
      const T ff = square(A.x.z);	    // f^2
      const T m  = A.x.x + A.y.y + A.z.z;
      const T c1 = (A.x.x*A.y.y + A.x.x*A.z.z + A.y.y*A.z.z)
	  - (dd + ee + ff);    // a*b + a*c + b*c - d^2 - e^2 - f^2
      const T c0 = A.z.z*dd + A.x.x*ee + A.y.y*ff - A.x.x*A.y.y*A.z.z
	  - T(2) * A.x.z*de;   // c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e)

      const T p = m*m - T(3)*c1;
      const T q = m*(p - (T(3)/T(2))*c1) - (T(27)/T(2))*c0;
      const T sqrt_p = sqrt(abs(p));

      T phi = T(27) * (T(0.25)*square(c1)*(p - c1) + c0*(q + T(27)/T(4)*c0));
      phi = (T(1)/T(3)) * atan2(sqrt(abs(phi)), q);

      constexpr T	M_SQRT3 = 1.73205080756887729352744634151;  // sqrt(3)
      const T	c = sqrt_p*cos(phi);
      const T	s = (T(1)/M_SQRT3)*sqrt_p*sin(phi);

      vec<T, 3>	w;
      w.y  = (T(1)/T(3))*(m - c);
      w.z  = w.y + s;
      w.x  = w.y + c;
      w.y -= s;

      return w;
  }

  template <class T> __device__ void
  tridiagonal33(const mat3x<T, 3>& A,
		mat3x<T, 3>& Qt, vec<T, 3>& d, vec<T, 3>& e)
  {
    // ----------------------------------------------------------------------
    // Reduces a symmetric 3x3 matrix to tridiagonal form by applying
    // (unitary) Householder transformations:
    //             [ d[0]  e[0]       ]
    //    Qt.A.Q = [ e[0]  d[1]  e[1] ]
    //             [       e[1]  d[2] ]
    // The function accesses only the diagonal and upper triangular parts of A.
    // The access is read-only.
    // ----------------------------------------------------------------------
      set_zero(Qt);
      Qt.x.x = Qt.y.y = Qt.z.z = T(1);

    // Bring first row and column to the desired form
      const T	h = square(A.x.y) + square(A.x.z);
      const T	g = (A.x.y > 0 ? -sqrt(h) : sqrt(h));
      e.x = g;

      T		f = g * A.x.y;
      T		omega = h - f;
      if (omega > T(0))
      {
	  const T	uy = A.x.y - g;
	  const T	uz = A.x.z;

	  omega = T(1) / omega;

	  f    = A.y.y * uy + A.y.z * uz;
	  T qy = omega * f;			// p
	  T K  = uy * f;			// u* A u
	  f    = A.y.z * uy + A.z.z * uz;
	  T qz = omega * f;			// p
	  K   += uz * f;			// u* A u

	  K  *= T(0.5) * square(omega);
	  qy -= K * uy;
	  qz -= K * uz;

	  d.x = A.x.x;
	  d.y = A.y.y - T(2)*qy*uy;
	  d.z = A.z.z - T(2)*qz*uz;

	// Store inverse Householder transformation in Q
	  f = omega * uy;
	  Qt.y.y -= f*uy;
	  Qt.z.y -= f*uz;
	  f = omega * uz;
	  Qt.y.z -= f*uy;
	  Qt.z.z -= f*uz;

	// Calculate updated A.y.z and store it in e.y
	  e.y = A.y.z - qy*uz - uy*qz;
      }
      else
      {
	  d.x = A.x.x;
	  d.y = A.y.y;
	  d.z = A.z.z;

	  e.y = A.y.z;
      }
  }

  namespace detail
  {
    template <class T> __device__ inline bool
    is_zero(T e, T g)
    {
	return abs(e) + g == g;
    }

    template <class T> __device__ T
    init_offdiagonal(T w, T w0, T w1, T e0)
    {
	const auto	t = (w1 - w0)/(e0 + e0);
	const auto	r = sqrt(square(t) + T(1));
	return w - w0 + e0/(t + (t > 0 ? r : -r));
    }

    template <size_t I, class T> __device__ inline void
    diagonalize(vec<T, 3>& w, vec<T, 3>& e,
		vec<T, 3>& q0, vec<T, 3>& q1, T& c, T& s)
    {
	const auto	x = val<I+1>(e);
	const auto	y = s*val<I>(e);
	const auto	z = c*val<I>(e);
	if (abs(x) > abs(y))
	{
	    const auto	t = y/x;
	    const auto	r = sqrt(square(t) + T(1));
	    val<I+1>(e) = x*r;
	    c 	      = T(1)/r;
	    s	      = c*t;
	}
	else
	{
	    const auto	t = x/y;
	    const auto	r = sqrt(square(t) + T(1));
	    val<I+1>(e) = y*r;
	    s	      = T(1)/r;
	    c	      = s*t;
	}

	const auto	v = s*(val<I>(w) - val<I+1>(w)) + T(2)*c*z;
	const auto	p = s*v;
	val<I  >(w) -= p;
	val<I+1>(w) += p;
	val<I  >(e)  = c*v - z;

      // Form eigenvectors
	const auto	q0_old = q0;
	(q0 *= c) -= s*q1;
	(q1 *= c) += s*q0_old;
    }

    template <class T> __device__ inline vec<T, 3>
    eigen_vector(vec<T, 3> a, vec<T, 3> b, T w)
    {
	a.x -= w;
	b.y -= w;
	return cross(a, b);
    }
  }	// namespace detail

  template <class T> __device__ bool
  qr33(const mat3x<T, 3>& A, mat3x<T, 3>& Qt, vec<T, 3>& w)
  {
    // Transform A to real tridiagonal form by the Householder method
      vec<T, 3>	e;	// The third element is used only as temporary
      device::tridiagonal33(A, Qt, w, e);

    // Calculate eigensystem of the remaining real symmetric tridiagonal matrix
    // with the QL method
    //
    // Loop over all off-diagonal elements
      for (int n = 0; n < 2; ++n)	// n = 0, 1
      {
	  for (int nIter = 0; ; )
	  {
	      int	i = n;
	      if (i == 0)
		  if (detail::is_zero(e.x, abs(w.x) + abs(w.y)))
		      e.x = T(0);
		  else
		      ++i;
	      if (i == 1)
		  if (detail::is_zero(e.y, abs(w.y) + abs(w.z)))
		      e.y = T(0);
		  else
		      ++i;
	      if (i == n)
		  break;

	      if (nIter++ >= 30)
		  return false;

	    // [n = 0]: i = 1, 2; [n = 1]: i = 2
	      if (n == 0)
		  if (i == 1)
		      e.y = detail::init_offdiagonal(w.y, w.x, w.y, e.x);
		  else
		      e.z = detail::init_offdiagonal(w.z, w.x, w.y, e.x);
	      else
		  e.z = detail::init_offdiagonal(w.z, w.y, w.z, e.y);

	      auto	s = T(1);
	      auto	c = T(1);
	    // [n = 0, i = 1]: i = 0; [n = 0, i = 2]: i = 1, 0
	    // [n = 1, i = 2]: i = 1
	      while (--i >= n)
		  if (i == 1)
		      detail::diagonalize<1>(w, e, Qt.y, Qt.z, c, s);
		  else
		      detail::diagonalize<0>(w, e, Qt.x, Qt.y, c, s);
	  }
      }

      return true;
  }

  template <class T> __device__  inline bool
  eigen33(const mat3x<T, 3>& A, mat3x<T, 3>& Qt, vec<T, 3>& w)
  {
      w = cardano(A);		// Calculate eigenvalues

      const auto	t     = min(min(abs(w.x), abs(w.y)), abs(w.z));
      const auto	u     = (t < T(1) ? t : square(t));
      const auto	error = T(256) * epsilon<T> * square(u);

    // 1st eigen vector
      Qt.x = detail::eigen_vector(A.x, A.y, w.x);
      auto	norm = dot(Qt.x, Qt.x);
      if (norm <= error)
	  return qr33(A, Qt, w);
      Qt.x *= rsqrt(norm);

    // 2nd eigen vector
      Qt.y = detail::eigen_vector(A.x, A.y, w.y);
      norm  = dot(Qt.y, Qt.y);
      if (norm <= error)
	  return qr33(A, Qt, w);
      Qt.y *= rsqrt(norm);

    // 3rd eigen vector
      Qt.z = cross(Qt.x, Qt.y);

      return true;
  }
}	// namespace device
#endif
}	// namespace cuda
}	// namespace TU
