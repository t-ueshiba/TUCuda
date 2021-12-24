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
/************************************************************************
*  struct BlockTraits							*
************************************************************************/
template <size_t BLOCK_DIM_X=32, size_t BLOCK_DIM_Y=16>
struct BlockTraits
{
  //! 1ブロックあたりのスレッド数(x方向)    
    constexpr static size_t	BlockDimX = BLOCK_DIM_X;
  //! 1ブロックあたりのスレッド数(y方向)    
    constexpr static size_t	BlockDimY = BLOCK_DIM_Y;
};
    
/************************************************************************
*  global functions							*
************************************************************************/
static inline size_t
gridDim(size_t dim, size_t blockDim)
{
    return (dim + blockDim - 1)/blockDim;
}
    
#if defined(__NVCC__)
inline std::ostream&
operator <<(std::ostream& out, const dim3& d)
{
    return out << '[' << d.x << ' ' << d.y << ' ' << d.z << ']';
}
    
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
  */
  template <class IN, class T> __device__ static inline void
  loadLine(const range<IN>& src, T dst[])
  {
      for (int tx = threadIdx.x; tx < src.size(); tx += blockDim.y)
	  dst[tx] = src[tx];
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

  //! スレッドブロックの横方向と縦方向にそれぞれ指定された長さを付加した領域をコピーする
  /*!
    \param src		コピー元の矩形領域の左上隅を指す反復子
    \param stride	コピー元の行を1つ進めるためのインクリメント数
    \param dst		コピー先の2次元配列
    \param dx		ブロック幅に付加される長さ
    \param dy		ブロック高に付加される長さ
  */
  template <class IN, class T, size_t W> __device__ static inline void
  loadTile(const range<range_iterator<IN> >& src, T dst[][W])
  {
      for (int ty = threadIdx.y; ty < src.size(); ty += blockDim.y)
      {
	  const auto	row = src[ty];
	  for (int tx = threadIdx.x; tx < row.size(); tx += blockDim.y)
	      dst[ty][tx] = row[tx];
      }
  }

  template <class IN, class T, size_t W> __device__ static inline void
  loadTileT(const range<range_iterator<IN> >& src, T dst[][W])
  {
      for (int ty = threadIdx.y; ty < src.size(); ty += blockDim.y)
      {
	  const auto	row = src[ty];
	  for (int tx = threadIdx.x; tx < row.size(); tx += blockDim.y)
	      dst[tx][ty] = row[tx];
      }
  }
}	// namespace device
#endif

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
*  subsample<BLOCK_TRAITS>(IN in, IN ie, OUT out)			*
************************************************************************/
//! CUDAによって2次元配列を水平／垂直方向それぞれ1/2に間引く．
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <class BLOCK_TRAITS=BlockTraits<>, class IN, class OUT> void
subsample(IN in, IN ie, OUT out)					;

#if defined(__NVCC__)
namespace device
{
  template <class IN, class OUT>
  __global__ static void
  subsample(range<range_iterator<IN> > in, range<range_iterator<OUT> > out)
  {
      const int	x = blockIdx.x*blockDim.x + threadIdx.x;
      const int	y = blockIdx.y*blockDim.y + threadIdx.y;

      if (2*y < in.size() && 2*x < in.begin().size())
	  out[y][x] = in[2*y][2*x];
  }
}	// namespace device

template <class BLOCK_TRAITS, class IN, class OUT> void
subsample(IN in, IN ie, OUT out)
{
    const auto	nrow = std::distance(in, ie);
    if (nrow < 2)
	return;

    const auto	ncol = TU::size(*in);
    if (ncol < 2)
	return;

    const dim3	threads(BLOCK_TRAITS::BlockDimX, BLOCK_TRAITS::BlockDimY);
    const dim3	blocks(gridDim(ncol/2, threads.x), gridDim(nrow/2, threads.y));
    device::subsample<<<blocks, threads>>>(cuda::make_range(in,  nrow),
					   cuda::make_range(out, nrow/2));
}
#endif

/************************************************************************
*  op3x3<BLOCK_TRAITS>(IN in, IN ie, OUT out, OP op)			*
************************************************************************/
//! CUDAによって2次元配列に対して3x3近傍演算を行う．
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
  \param op	3x3近傍演算子
*/
template <class BLOCK_TRAITS=BlockTraits<>, class IN, class OUT, class OP>
void	op3x3(IN in, IN ie, OUT out, OP op)				;

#if defined(__NVCC__)
namespace device
{
  template <class BLOCK_TRAITS, class IN, class OUT, class OP>
  __global__ static void
  op3x3(range<range_iterator<IN> > in, range<range_iterator<OUT> > out, OP op)
  {
      using	value_type = typename std::iterator_traits<IN>::value_type;

      const int	x0 = __mul24(blockIdx.x, blockDim.x);
      const int	y0 = __mul24(blockIdx.y, blockDim.y);

    // 原画像のブロック内部およびその外枠1画素分を共有メモリに転送
      __shared__ value_type	in_s[BLOCK_TRAITS::BlockDimY + 2]
				    [BLOCK_TRAITS::BlockDimX + 2 + 1];
      loadTile(slice(in.cbegin(),
		     y0, ::min(blockDim.y + 2, in.size() - y0),
		     x0, ::min(blockDim.x + 2, in.begin().size() - x0)),
	       in_s);
      __syncthreads();

    // 共有メモリに保存した原画像から現在画素に対するフィルタ出力を計算
      const int	tx = threadIdx.x;
      const int ty = threadIdx.y;
      const int	x  = x0 + tx;
      const int	y  = y0 + ty;
      if (y + 2 < in.size() && x + 2 < in.begin().size())
	  out[y + 1][x + 1] = op(in_s[ty]     + tx,
				 in_s[ty + 1] + tx,
				 in_s[ty + 2] + tx);
  }
}	// namespace device

template <class BLOCK_TRAITS, class IN, class OUT, class OP> void
op3x3(IN in, IN ie, OUT out, OP op)
{
    const auto	nrow = std::distance(in, ie);
    if (nrow < 1)
	return;

    const auto	ncol = TU::size(*in);
    if (ncol < 1)
	return;

    const dim3	threads(BLOCK_TRAITS::BlockDimX, BLOCK_TRAITS::BlockDimY);
    const dim3	blocks(gridDim(ncol, threads.x), gridDim(nrow, threads.y));
    device::op3x3<BLOCK_TRAITS><<<blocks, threads>>>(
	cuda::make_range(in, nrow), cuda::make_range(out, nrow), op);
}
#endif

/************************************************************************
*  suppressNonExtrema<BLOCK_TRAITS>(					*
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
template <class BLOCK_TRAITS=BlockTraits<>, class IN, class OUT, class OP> void
suppressNonExtrema3x3(
    IN in, IN ie, OUT out, OP op,
    typename std::iterator_traits<IN>::value_type::value_type nulval=0)	;

#if defined(__NVCC__)
namespace device
{
  template <class BLOCK_TRAITS, class IN, class OUT, class OP>
  __global__ static void
  extrema3x3(range<range_iterator<IN> > in, range<range_iterator<OUT> > out,
	     OP op, typename std::iterator_traits<IN>::value_type nulval)
  {
      using	value_type = typename std::iterator_traits<IN>::value_type;

      const int	x0 = 2*__mul24(blockIdx.x, blockDim.x);
      const int	y0 = 2*__mul24(blockIdx.y, blockDim.y);

      __shared__ value_type	in_s[2*BLOCK_TRAITS::BlockDimY + 2]
				    [2*BLOCK_TRAITS::BlockDimX + 3];
      loadTile(slice(in.cbegin(),
		     y0, ::min(2*blockDim.y + 2, in.size() - y0),
		     x0, ::min(2*blockDim.x + 2, in.begin().size() - x0)),
	       in_s);
      __syncthreads();

    // このスレッドの処理対象である2x2ウィンドウ中で最大/最小となる画素の座標を求める．
      const int	x = 1 + 2*threadIdx.x;
      const int	y = 1 + 2*threadIdx.y;

      if (y0 + y + 1 < in.size() && x0 + x + 1 < in.begin().size())
      {
	//const int	i01 = (op(in_s[y    ][x], in_s[y    ][x + 1]) ? 0 : 1);
	//const int	i23 = (op(in_s[y + 1][x], in_s[y + 1][x + 1]) ? 2 : 3);
	  const int	i01 = op(in_s[y    ][x + 1], in_s[y    ][x]);
	  const int	i23 = op(in_s[y + 1][x + 1], in_s[y + 1][x]) + 2;
	  const int	iex = (op(in_s[y    ][x + i01],
				  in_s[y + 1][x + (i23 & 0x1)]) ? i01 : i23);
	  const int	xx  = x + (iex & 0x1);		// 最大/最小点のx座標
	  const int	yy  = y + (iex >> 1);		// 最大/最小点のy座標

	// 最大/最小画素が，残り5つの近傍点よりも大きい/小さいか調べる．
	//const int	dx  = (iex & 0x1 ? 1 : -1);
	//const int	dy  = (iex & 0x2 ? 1 : -1);
	  const int	dx  = ((iex & 0x1) << 1) - 1;
	  const int	dy  = (iex & 0x2) - 1;
	  auto		val = in_s[yy][xx];
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
	  in_s[yy   ][xx   ] = val;		// 極値または非極値
	  __syncthreads();

	// (2*blockDim.x)x(2*blockDim.y) の矩形領域に共有メモリ領域をコピー．
	  out[y0 + y - 1][x0 + x - 1] = in_s[y    ][x    ];
	  out[y0 + y - 1][x0 + x    ] = in_s[y    ][x + 1];
	  out[y0 + y    ][x0 + x - 1] = in_s[y + 1][x    ];
	  out[y0 + y    ][x0 + x    ] = in_s[y + 1][x + 1];
      }
  }
}	// namespace device

template <class BLOCK_TRAITS, class IN, class OUT, class OP> void
suppressNonExtrema3x3(
    IN in, IN ie, OUT out, OP op,
    typename std::iterator_traits<IN>::value_type::value_type nulval)
{
    const int	nrow = std::distance(in, ie);
    if (nrow < 3)
	return;

    const int	ncol = TU::size(*in);
    if (ncol < 3)
	return;

    const dim3	threads(BLOCK_TRAITS::BlockDimX, BLOCK_TRAITS::BlockDimY);
    const dim3	blocks(gridDim((ncol - 1)/2, threads.x),
		       gridDim((nrow - 1)/2, threads.y));
    device::extrema3x3<BLOCK_TRAITS><<<blocks, threads>>>(
	cuda::make_range(in,  nrow), cuda::make_range(out, nrow), op, nulval);
}
#endif

/************************************************************************
*  transpose<BLOCK_DIM>(IN in, IN ie, OUT out)				*
************************************************************************/
//! CUDAによって2次元配列の転置処理を行う．
/*!
  \param in	入力2次元配列の最初の行を指す反復子
  \param ie	入力2次元配列の最後の次の行を指す反復子
  \param out	出力2次元配列の最初の行を指す反復子
*/
template <size_t BLOCK_DIM=32, class IN, class OUT> void
transpose(IN in, IN ie, OUT out)					;

#if defined(__NVCC__)
namespace device
{
  template <size_t BLOCK_DIM, class IN, class OUT>
  __global__ static void
  transpose(range<range_iterator<IN> > in, range<range_iterator<OUT> > out)
  {
      using	value_type = typename std::iterator_traits<IN>::value_type;

      __shared__ value_type	tile[BLOCK_DIM][BLOCK_DIM + 1];

      const int	x0 = __mul24(blockIdx.x, blockDim.x);
      const int	y0 = __mul24(blockIdx.y, blockDim.y);

      tile[threadIdx.y][threadIdx.x] = in[y0 + threadIdx.y][x0 + threadIdx.x];
      __syncthreads();

      out[x0 + threadIdx.y][y0 + threadIdx.x] = tile[threadIdx.x][threadIdx.y];
  }
}	// namespace device

namespace detail
{
  template <size_t BLOCK_DIM, class IN, class OUT> static void
  transpose(IN in, IN ie, OUT out, size_t i, size_t j)
  {
      size_t	r = std::distance(in, ie);
      if (r < 1)
	  return;

      size_t	c = TU::size(*in) - j;
      if (c < 1)
	  return;

      const auto	blockDim = std::min({BLOCK_DIM, r, c});
      const dim3	threads(blockDim, blockDim);
      const dim3	blocks(c/threads.x, r/threads.y);
      cuda::device::transpose<BLOCK_DIM><<<blocks, threads>>>(
	  cuda::slice(in, 0, r, j, c), cuda::slice(out, 0, c, i, r));	// 左上

      r = blocks.y*threads.y;
      c = blocks.x*threads.x;

      auto	in_n = in;
      std::advance(in_n, r);
      auto	out_n = out;
      std::advance(out_n, c);
      transpose<BLOCK_DIM>(in,   in_n, out_n, i,     j + c);	// 右上
      transpose<BLOCK_DIM>(in_n, ie,   out,   i + r, j);	// 下
  }
}	// namesapce detail

template <size_t BLOCK_DIM, class IN, class OUT> inline void
transpose(IN in, IN ie, OUT out)
{
    detail::transpose<BLOCK_DIM>(in, ie, out, 0, 0);
}
#endif

/************************************************************************
*  transform2<BLOCK_TRAITS>(IN in, IN ie, OUT out, OP op)		*
************************************************************************/
template <class BLOCK_TRAITS=BlockTraits<>, class IN, class OUT, class OP> void
transform2(IN in, IN ie, OUT out, OP op)				;

#if defined(__NVCC__)
namespace device
{
  namespace detail
  {
      template <class OP, class T>
      static auto	check_unary(OP op, T arg)
			    -> decltype(op(arg), std::true_type());
      template <class OP, class T>
      static auto	check_unary(OP op, T arg)
			    -> decltype(op(0, 0, arg), std::false_type());
      template <class OP, class T>
      using is_unary	= decltype(check_unary(std::declval<OP>(),
					       std::declval<T>()));

      template <class OP, class T,
		std::enable_if_t<is_unary<OP, T>::value>* = nullptr>
      __device__ decltype(auto)
      apply(OP&& op, int x, int y, T&& arg)
      {
	  return op(std::forward<T>(arg));
      }

      template <class OP, class T,
		std::enable_if_t<!is_unary<OP, T>::value>* = nullptr>
      __device__ decltype(auto)
      apply(OP&& op, int x, int y, T&& arg)
      {
	  return op(x, y, std::forward<T>(arg));
      }
  }
    
  template <class IN, class OUT, class OP> __global__ static void
  transform2(range<range_iterator<IN> > in,
	     range<range_iterator<OUT> > out, OP op)
  {
      const int	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
      const int	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

      if (y < in.size() && x < in.begin().size())
	  out[y][x] = detail::apply(op, x, y, in[y][x]);
  }
}	// namespace device

template <class BLOCK_TRAITS, class IN, class OUT, class OP> void
transform2(IN in, IN ie, OUT out, OP op)
{
    const auto	nrow = std::distance(in, ie);
    if (nrow < 1)
	return;

    const auto	ncol = TU::size(*in);
    if (ncol < 1)
	return;

    const dim3	threads(BLOCK_TRAITS::BlockDimX, BLOCK_TRAITS::BlockDimY);
    const dim3	blocks(gridDim(ncol, threads.x), gridDim(nrow, threads.y));
    device::transform2<<<blocks, threads>>>(cuda::make_range(in,  nrow),
					    cuda::make_range(out, nrow), op);
}
#endif
}	// namespace cuda
}	// namespace TU
