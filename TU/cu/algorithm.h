// Software License Agreement (BSD License)
//
// Copyright (c) 2021, National Institute of Advanced Industrial Science and Technology (AIST)
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//  * Neither the name of National Institute of Advanced Industrial
//    Science and Technology (AIST) nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Toshio Ueshiba
//
/*!
  \file		algorithm.h
  \brief	各種アルゴリズムの定義と実装
*/
#pragma once

#include "TU/cu/allocator.h"
#include "TU/cu/iterator.h"

#define gpuCheckError(err)	TU::cu::checkError(err, __FILE__, __LINE__)
#define gpuCheckLastError()	TU::cu::checkError(cudaGetLastError(),	\
 						   __FILE__, __LINE__)
#define gpuCheckAsyncError()	TU::cu::checkError(cudaDeviceSynchronize(), \
 						   __FILE__, __LINE__)

namespace TU
{
namespace cu
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
  //! 1ブロックあたりのスレッド数(1次元)
    constexpr static size_t	BlockDim  = BlockDimX * BlockDimY;
};

/************************************************************************
*  global functions							*
************************************************************************/
inline size_t
divUp(size_t dim, size_t blockDim)
{
    return (dim + blockDim - 1)/blockDim;
}

#if defined(__NVCC__)
inline std::ostream&
operator <<(std::ostream& out, const dim3& d)
{
    return out << '[' << d.x << ' ' << d.y << ' ' << d.z << ']';
}

inline void
checkError(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess)
    {
	std::cerr << "CUDA Runtime Error at: " << file << ':' << line
		  << '[' << cudaGetErrorString(err) << ']' << std::endl;
	std::exit(EXIT_FAILURE);
    }
}

//! デバイス関数を納める名前空間
namespace device
{
  /**********************************************************************
  *  __device__ functions						*
  **********************************************************************/
  //! スレッドブロック中の1次元領域をコピーする
  /*!
    \param src		コピー元の1次元領域
    \param dst		コピー先の1次元配列
  */
  template <class IN, class T> __device__ __forceinline__ static void
  loadLine(const range<IN>& src, T dst[])
  {
      for (int tx = threadIdx.x; tx < src.size(); tx += blockDim.x)
	  dst[tx] = src[tx];
  }

  //! スレッドブロック中の2次元領域をコピーする
  /*!
    \param src		コピー元の2次元領域
    \param dst		コピー先の2次元配列
  */
  template <class IN, class T, size_t W> __device__ __forceinline__ static void
  loadTile(const range<range_iterator<IN> >& src, T dst[][W])
  {
      for (int ty = threadIdx.y; ty < src.size(); ty += blockDim.y)
      {
	  const auto	row = src[ty];
	  for (int tx = threadIdx.x; tx < row.size(); tx += blockDim.x)
	      dst[ty][tx] = row[tx];
      }
  }

  //! スレッドブロック中の2次元領域を転置コピーする
  /*!
    \param src		コピー元の2次元領域
    \param dst		コピー先の2次元配列
  */
  template <class IN, class T, size_t W> __device__ __forceinline__ static void
  loadTileT(const range<range_iterator<IN> >& src, T dst[][W])
  {
      for (int ty = threadIdx.y; ty < src.size(); ty += blockDim.y)
      {
	  const auto	row = src[ty];
	  for (int tx = threadIdx.x; tx < row.size(); tx += blockDim.x)
	      dst[tx][ty] = row[tx];
      }
  }

  template <class T, size_t H, size_t W> __device__ __forceinline__ static void
  reverse(const T src[H][W], T dst[][W])
  {
      for (int ty = threadIdx.y; ty < H; ty += blockDim.y)
	  for (int tx = threadIdx.x; tx < W; tx += blockDim.x)
	      dst[ty][tx] = src[ty][W - 1 - tx];
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
template <class BLOCK_TRAITS=BlockTraits<>, class IN, class OUT, class OP> void
subsample(IN in, IN ie, OUT out, OP op)					;

#if defined(__NVCC__)
namespace device
{
template <class BLOCK_TRAITS, class IN, class OUT, class OP>
  __global__ static void
  subsample(range<range_iterator<IN> > in,
	    range<range_iterator<OUT> > out, OP op)
  {
      using	value_type = typename std::iterator_traits<IN>::value_type;

      constexpr int	M	= OP::OperatorSizeX;
      constexpr int	N	= OP::OperatorSizeY;
      constexpr int	Stride	= 2*BLOCK_TRAITS::BlockDimX + M - 1;
      constexpr int	OffsetY	= (N - 1)/2;
      constexpr int	OffsetX	= (M - 1)/2;

      const int	x0 = __mul24(blockIdx.x, blockDim.x);
      const int	y0 = __mul24(blockIdx.y, blockDim.y);
      const int	xs = ::max(2*x0 - OffsetX, 0);
      const int	ys = ::max(2*y0 - OffsetY, 0);

    // 原画像のブロック内部およびその外枠1画素分を共有メモリに転送
      __shared__ value_type in_s[2*BLOCK_TRAITS::BlockDimY + N - 1][Stride];
      loadTile(slice(in.cbegin(),
		     ys, ::min(int(2*blockDim.y + N - 1), in.size() - ys),
		     xs, ::min(int(2*blockDim.x + M - 1),
			       in.begin().size() - xs)),
	       in_s);
      __syncthreads();

      const int	x = x0 + threadIdx.x;
      const int	y = y0 + threadIdx.y;
      if (2*y < in.size() && 2*x < in.begin().size())
	  out[y][x] = op(2*y - ys, 2*x - xs,
			 in.size() - ys, in.begin().size() - xs, in_s);
  }
}	// namespace device

template <class BLOCK_TRAITS, class IN, class OUT, class OP> void
subsample(IN in, IN ie, OUT out, OP op)
{
    using	std::size;

    const int	nrow = std::distance(in, ie);
    if (nrow < 2)
	return;

    const int	ncol = size(*in);
    if (ncol < 2)
	return;

    const dim3	threads(BLOCK_TRAITS::BlockDimX, BLOCK_TRAITS::BlockDimY);
    const dim3	blocks(divUp(ncol/2, threads.x), divUp(nrow/2, threads.y));
    device::subsample<BLOCK_TRAITS><<<blocks, threads>>>(
	cu::make_range(in,  nrow), cu::make_range(out, nrow/2), op);
    gpuCheckLastError();
}
#endif

/************************************************************************
*  opNxM<BLOCK_TRAITS>(IN in, IN ie, OUT out, OP op)			*
************************************************************************/
template <class BLOCK_TRAITS=BlockTraits<>, class IN, class OUT, class OP>
void	opNxM(IN in, IN ie, OUT out, OP op)				;

#if defined(__NVCC__)
namespace device
{
  template <class BLOCK_TRAITS, class IN, class OUT, class OP>
  __global__ static void
  opNxM(range<range_iterator<IN> > in, range<range_iterator<OUT> > out, OP op)
  {
      using	value_type = typename std::iterator_traits<IN>::value_type;

      constexpr int	M	= OP::OperatorSizeX;
      constexpr int	N	= OP::OperatorSizeY;
      constexpr int	Stride	= BLOCK_TRAITS::BlockDimX + M - 1;
      constexpr int	OffsetX = (M - 1)/2;
      constexpr int	OffsetY = (N - 1)/2;

      const int	x0 = __mul24(blockIdx.x, blockDim.x);
      const int	y0 = __mul24(blockIdx.y, blockDim.y);
      const int	xs = ::max(x0 - OffsetX, 0);
      const int	ys = ::max(y0 - OffsetY, 0);

    // 原画像のブロック内部およびその外枠1画素分を共有メモリに転送
      __shared__ value_type	in_s[BLOCK_TRAITS::BlockDimY + N - 1][Stride];
      loadTile(slice(in.cbegin(),
		     ys, ::min(int(blockDim.y + N - 1), in.size() - ys),
		     xs, ::min(int(blockDim.x + M - 1),
			       in.begin().size() - xs)),
	       in_s);
      __syncthreads();

    // 共有メモリに保存した原画像から現在画素に対するフィルタ出力を計算
      const int	x = x0 + threadIdx.x;
      const int	y = y0 + threadIdx.y;
      if (y < in.size() && x < in.begin().size())
	  out[y][x] = op(y - ys, x - xs,
			 in.size() - ys, in.begin().size() - xs, in_s);
  }
}	// namespace device

template <class BLOCK_TRAITS, class IN, class OUT, class OP> void
opNxM(IN in, IN ie, OUT out, OP op)
{
    using	std::size;

    const int	nrow = std::distance(in, ie);
    if (nrow < 1)
	return;

    const int	ncol = size(*in);
    if (ncol < 1)
	return;

    const dim3	threads(BLOCK_TRAITS::BlockDimX, BLOCK_TRAITS::BlockDimY);
    const dim3	blocks(divUp(ncol, threads.x), divUp(nrow, threads.y));
    device::opNxM<BLOCK_TRAITS><<<blocks, threads>>>(
	cu::make_range(in, nrow), cu::make_range(out, nrow), op);
    gpuCheckLastError();
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
      using	std::size;

      size_t	r = std::distance(in, ie);
      if (r < 1)
	  return;

      size_t	c = size(*in) - j;
      if (c < 1)
	  return;

      const auto	blockDim = std::min({BLOCK_DIM, r, c});
      const dim3	threads(blockDim, blockDim);
      const dim3	blocks(c/threads.x, r/threads.y);
      cu::device::transpose<BLOCK_DIM><<<blocks, threads>>>(
	  cu::slice(in, 0, r, j, c), cu::slice(out, 0, c, i, r));	// 左上
      gpuCheckLastError();

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
  template <class IN, class OUT, class OP> __global__ static void
  transform2(range<range_iterator<IN> > in,
	     range<range_iterator<OUT> > out, OP op, std::true_type)
  {
      const int	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
      const int	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

      if (y < in.size() && x < in.begin().size())
	  out[y][x] = op(in[y][x]);
  }

  template <class IN, class OUT, class OP> __global__ static void
  transform2(range<range_iterator<IN> > in,
	     range<range_iterator<OUT> > out, OP op, std::false_type)
  {
      const int	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
      const int	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

      if (y < in.size() && x < in.begin().size())
	  out[y][x] = op(x, y, in[y][x]);
  }
}	// namespace device

namespace detail
{
  template <class OP, class T>
  static auto	check_unarity(OP op, T arg)
		    -> decltype(op(arg), std::true_type());
  template <class OP, class T>
  static auto	check_unarity(OP op, T arg)
		    -> decltype(op(0, 0, arg), std::false_type());
  template <class OP, class T>
  using is_unary = decltype(check_unarity(std::declval<OP>(),
					  std::declval<T>()));
}

template <class BLOCK_TRAITS, class IN, class OUT, class OP> void
transform2(IN in, IN ie, OUT out, OP op)
{
    using std::size;
    using value_type	= typename std::iterator_traits<IN>::value_type
							   ::value_type;

    const int	nrow = std::distance(in, ie);
    if (nrow < 1)
	return;

    const int	ncol = size(*in);
    if (ncol < 1)
	return;

    const dim3	threads(BLOCK_TRAITS::BlockDimX, BLOCK_TRAITS::BlockDimY);
    const dim3	blocks(divUp(ncol, threads.x), divUp(nrow, threads.y));
    device::transform2<<<blocks, threads>>>(
	cu::make_range(in,  nrow), cu::make_range(out, nrow), op,
	std::integral_constant<bool,
			       detail::is_unary<OP, value_type>::value>());
    gpuCheckLastError();
}
#endif

/************************************************************************
*  copy2<BLOCK_TRAITS>(IN in, IN ie, OUT out)				*
************************************************************************/
template <class BLOCK_TRAITS=BlockTraits<>, class IN, class OUT> void
copy2(IN in, IN ie, OUT out)
{
    using value_type	= typename std::iterator_traits<IN>::value_type
							   ::value_type;

    transform2<BLOCK_TRAITS>(in, ie, out,
			     [] __host__ __device__ (const value_type& val)
			     { return val; });
}

/************************************************************************
*  generate2<BLOCK_TRAITS>(OUT out, OUT oe, GEN gen)			*
************************************************************************/
template <class BLOCK_TRAITS=BlockTraits<>, class OUT, class GEN> void
generate2(OUT out, OUT oe, GEN gen)					;

#if defined(__NVCC__)
namespace device
{
  template <class OUT, class GEN> __global__ static void
  generate2(range<range_iterator<OUT> > out, GEN gen, std::true_type)
  {
      const int	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
      const int	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

      if (y < out.size() && x < out.begin().size())
	  out[y][x] = gen();
  }

  template <class OUT, class GEN> __global__ static void
  generate2(range<range_iterator<OUT> > out, GEN gen, std::false_type)
  {
      const int	x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
      const int	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;

      if (y < out.size() && x < out.begin().size())
	  out[y][x] = gen(y, x);
  }
}	// namespace cu

namespace detail
{
  template <class GEN>
  static auto	check_noargs(GEN gen) -> decltype(gen(), std::true_type());
  template <class GEN>
  static auto	check_noargs(GEN gen) -> decltype(gen(0, 0), std::false_type());
  template <class GEN>
  using noargs = decltype(check_noargs(std::declval<GEN>()));
}	// namespace detail

template <class BLOCK_TRAITS, class OUT, class GEN> void
generate2(OUT out, OUT oe, GEN gen)
{
    using	std::size;

    const int	nrow = std::distance(out, oe);
    if (nrow < 1)
	return;

    const int	ncol = size(*out);
    if (ncol < 1)
	return;

    const dim3	threads(BLOCK_TRAITS::BlockDimX, BLOCK_TRAITS::BlockDimY);
    const dim3	blocks(divUp(ncol, threads.x), divUp(nrow, threads.y));
    device::generate2<<<blocks, threads>>>(
    	cu::make_range(out, nrow), gen,
    	std::integral_constant<bool, detail::noargs<GEN>::value>());
    gpuCheckLastError();
}
#endif

/************************************************************************
*  fill2<BLOCK_TRAITS>(OUT out, OUT oe, T val)				*
************************************************************************/
template <class BLOCK_TRAITS=BlockTraits<>, class OUT, class T> void
fill2(OUT out, OUT oe, T val)
{
    generate2<BLOCK_TRAITS>(out, oe,
			    [val] __host__ __device__ (){ return val; });
}

}	// namespace cu
}	// namespace TU
