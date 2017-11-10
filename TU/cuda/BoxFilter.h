/*
 *  $Id: BoxFilter.h 1962 2016-03-22 02:56:32Z ueshiba $
 */
/*!
  \file		BoxFilter.h
  \brief	boxフィルタの定義と実装
*/ 
#ifndef TU_CUDA_BOXFILTER_H
#define TU_CUDA_BOXFILTER_H

#include "TU/cuda/Array++.h"
#include "TU/cuda/algorithm.h"

namespace TU
{
namespace cuda
{
/************************************************************************
*  class BoxFilter2<T, WMAX>						*
************************************************************************/
//! CUDAによる2次元boxフィルタを表すクラス
template <class T, size_t WMAX=32>
class BoxFilter2
{
  public:
  //! CUDAによる2次元boxフィルタを生成する．
  /*!
    \param rowWinSize	boxフィルタのウィンドウの行幅(高さ)
    \param colWinSize	boxフィルタのウィンドウの列幅(幅)
   */	
		BoxFilter2(size_t rowWinSize, size_t colWinSize)
		    :_rowWinSize(rowWinSize), _colWinSize(colWinSize)
		{
		    if (_rowWinSize > WMAX || _colWinSize > WMAX)
			throw std::runtime_error("Too large window size!");
		}

  //! boxフィルタのウィンドウ行幅(高さ)を返す．
  /*!
    \return	boxフィルタのウィンドウの行幅
   */
    size_t	rowWinSize()			const	{ return _rowWinSize; }

  //! boxフィルタのウィンドウ列幅(幅)を返す．
  /*!
    \return	boxフィルタのウィンドウの列幅
   */
    size_t	colWinSize()			const	{ return _colWinSize; }

  //! boxフィルタのウィンドウの行幅(高さ)を設定する．
  /*!
    \param rowWinSize	boxフィルタのウィンドウの行幅
    \return		このboxフィルタ
   */
    BoxFilter2&	setRowWinSize(size_t rowWinSize)
		{
		    _rowWinSize = rowWinSize;
		    return *this;
		}

  //! boxフィルタのウィンドウの列幅(幅)を設定する．
  /*!
    \param colWinSize	boxフィルタのウィンドウの列幅
    \return		このboxフィルタ
   */
    BoxFilter2&	setColWinSize(size_t colWinSize)
		{
		    _colWinSize = colWinSize;
		    return *this;
		}
    
  //! 与えられた2次元配列とこのフィルタの畳み込みを行う
  /*!
    \param row	入力2次元データ配列の先頭行を指す反復子
    \param rowe	入力2次元データ配列の末尾の次の行を指す反復子
    \param rowO	出力2次元データ配列の先頭行を指す反復子
  */
    template <class ROW, class ROW_O>
    void	convolve(ROW row, ROW rowe, ROW_O rowO)		const	;

  //! 2組の2次元配列間の相違度とこのフィルタの畳み込みを行う
  /*!
    \param rowL			左入力2次元データ配列の先頭行を指す反復子
    \param rowLe		左入力2次元データ配列の末尾の次の行を指す反復子
    \param rowR			右入力2次元データ配列の先頭行を指す反復子
    \param rowO			出力2次元相違度配列の先頭行を指す反復子
    \param op			左右の画素間の相違度
    \param disparitySearchWidth	視差の探索幅
  */
    template <class ROW, class ROW_O, class OP>
    void	convolve(ROW rowL, ROW rowLe, ROW rowR, ROW_O rowO,
			 OP op, size_t disparitySearchWidth)	const	;

  public:
    static constexpr size_t	winSizeMax = WMAX;
    static constexpr size_t	BlockDim   = 16;
    static constexpr size_t	BlockDimX  = BlockDim;
    static constexpr size_t	BlockDimY  = 16;
    
  private:
    size_t		_rowWinSize;
    size_t		_colWinSize;
    mutable Array2<T>	_buf;
    mutable Array3<T>	_buf3;
};

#if defined(__NVCC__)
namespace device
{
/************************************************************************
*  __global__ functions							*
************************************************************************/
//! スレッドブロックの縦方向にフィルタを適用する
/*!
  sliding windowを使用するが threadIdx.y が0のスレッドしか仕事をしないので
  ウィンドウ幅が大きいときのみ高効率．また，結果は転置して格納される．
  \param col		入力2次元配列の左上隅を指す反復子
  \param colO		出力2次元配列の左上隅を指す反復子
  \param winSize	boxフィルタのウィンドウの行幅(高さ)
  \param strideI	入力2次元配列の行を1つ進めるためにインクリメントするべき要素数
  \param strideO	出力2次元配列の行を1つ進めるためにインクリメントするべき要素数
*/
template <size_t WMAX, class COL, class COL_O> __global__ static void
box_filter(COL col, COL_O colO, int winSize, int strideI, int strideO)
{
    using	out_type = typename std::iterator_traits<COL_O>::value_type;
    
    constexpr auto	BlockDim = BoxFilter2<out_type, WMAX>::BlockDim;
    __shared__ out_type	in_s[BlockDim + WMAX - 1][BlockDim + 1];
    __shared__ out_type	out_s[BlockDim][BlockDim + 1];

    const auto	x0 = __mul24(blockIdx.x, blockDim.x);	// ブロック左上隅
    const auto	y0 = __mul24(blockIdx.y, blockDim.y);	// ブロック左上隅

    loadTileV(col + __mul24(y0, strideI) + x0, strideI, in_s, winSize - 1);
    __syncthreads();
    
    if (threadIdx.y == 0)
    {
      // 各列を並列に縦方向積算
	const auto	tx = threadIdx.x;
	
	out_s[0][tx] = in_s[0][tx];
	for (int y = 1; y != winSize; ++y)
	    out_s[0][tx] += in_s[y][tx];

	for (int y = 1; y != blockDim.y; ++y)
	    out_s[y][tx] = out_s[y-1][tx]
			 + in_s[y-1+winSize][tx] - in_s[y-1][tx];
    }
    __syncthreads();

  // 結果を転置して格納
    if (blockDim.x == blockDim.y)
	colO[__mul24(x0 + threadIdx.y, strideO) + y0 + threadIdx.x] =
	    out_s[threadIdx.x][threadIdx.y];
    else
	colO[__mul24(x0 + threadIdx.x, strideO) + y0 + threadIdx.y] =
	    out_s[threadIdx.y][threadIdx.x];
}

template <size_t WMAX, class COL, class COL_O, class OP> __global__ static void
box_filterV(COL colL, COL colR, int nrows, COL_O colO, OP op, int rowWinSize,
	    int strideL, int strideR, int strideD, int strideXD)
{
    using	out_type = typename std::iterator_traits<COL_O>::value_type;

    constexpr auto	BlockDimX = BoxFilter2<out_type, WMAX>::BlockDimX;
    constexpr auto	BlockDimY = BoxFilter2<out_type, WMAX>::BlockDimY;

    __shared__ out_type	inL_s[BlockDimY];
    __shared__ out_type	inR_s[BlockDimY + BlockDimX - 1];
    __shared__ out_type	mid_s[WMAX][BlockDimY][BlockDimX];

    const auto	d0 = __mul24(blockIdx.x, blockDim.x);	// 視差
    const auto	x0 = __mul24(blockIdx.y, blockDim.y);	// 列

    colL += x0;
    colR += (x0 + d0);
    colO += (__mul24(x0 + threadIdx.y, strideD) + d0 + threadIdx.x);
    
  // 最初のwinSize画素分の相違度を計算してvalに積算
    out_type	val = 0;
    for (int i = 0; i != rowWinSize; ++i)
    {
	if (threadIdx.y == 0)
	{
	    if (threadIdx.x < blockDim.y)
		inL_s[threadIdx.x] = colL[threadIdx.x];
	    loadLine(colR, inR_s, blockDim.y - 1);

	    colL += strideL;
	    colR += strideR;
	}
	__syncthreads();
	
	val += (mid_s[i][threadIdx.y][threadIdx.x]
		= op(inL_s[threadIdx.y], inR_s[threadIdx.y + threadIdx.x]));
    }
    *colO = val;
    
  // 逐次的にvalを更新して出力
    for (int y = 1, i = 0; y < nrows; ++y)
    {
	val -= mid_s[i][threadIdx.y][threadIdx.x];

	if (threadIdx.y == 0)
	{
	    if (threadIdx.x < blockDim.y)
		inL_s[threadIdx.x] = colL[threadIdx.x];
	    loadLine(colR, inR_s, blockDim.y - 1);

	    colL += strideL;
	    colR += strideR;
	}
	__syncthreads();

	*(colO += strideXD) = (val += (mid_s[i][threadIdx.y][threadIdx.x]
				       = op(inL_s[threadIdx.y],
					    inR_s[threadIdx.y + threadIdx.x])));

	if (++i == rowWinSize)
	    i = 0;
    }
}

template <size_t WMAX, class COL, class COL_O> __global__ static void
box_filterH(COL col, int ncols, COL_O colO,
	    int colWinSize, int strideXD, int strideD)
{
    using	out_type = typename std::iterator_traits<COL_O>::value_type;

    constexpr auto	BlockDimX = BoxFilter2<out_type, WMAX>::BlockDimX;
    constexpr auto	BlockDimY = BoxFilter2<out_type, WMAX>::BlockDimY;
    
    __shared__ out_type	mid_s[WMAX][BlockDimY][BlockDimX];

    const auto	d = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// 視差
    const auto	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;	// 行

    col  += (__mul24(y, strideXD) + d);
    colO += (__mul24(y, strideXD) + d);
    
    auto	val = (mid_s[0][threadIdx.y][threadIdx.x] = *col);
    for (int i = 0; ++i < colWinSize; )
	val += (mid_s[i][threadIdx.y][threadIdx.x] = *(col += strideD));
    *colO = val;
    
    for (int x = 1, i = 0; x < ncols; ++x)
    {
	val -= mid_s[i][threadIdx.y][threadIdx.x];
	*(colO += strideD) = (val += (mid_s[i][threadIdx.y][threadIdx.x]
				      = *(col += strideD)));
	
	if (++i == colWinSize)
	    i = 0;
    }
}
}	// namespace device
/************************************************************************
*  class BoxFilter2							*
************************************************************************/
template <class T, size_t WMAX> template <class ROW, class ROW_O> void
BoxFilter2<T, WMAX>::convolve(ROW row, ROW rowe, ROW_O rowO) const
{
    auto	nrows = std::distance(row, rowe);
    if (nrows < _rowWinSize)
	return;
    
    auto	ncols = std::distance(row->cbegin(), row->cend());
    if (ncols < _colWinSize)
	return;

    ncols -= (_colWinSize - 1);

    _buf.resize(32, ncols, nrows);
    
    const auto	strideI = stride(row);
    const auto	strideB = _buf.stride();
    const auto	strideO = stride(rowO);

  // ---- 縦方向積算 ----
  // 左上
    dim3	threads(BlockDim, BlockDim);
    dim3	blocks(ncols/threads.x, nrows/threads.y);
    device::box_filter<WMAX><<<blocks, threads>>>(row->cbegin().get(),
						  _buf[0].begin().get(),
						  _rowWinSize,
						  strideI, strideB);
  // 右上
    auto	x = blocks.x*threads.x;
    threads.x = ncols%threads.x;
    blocks.x  = 1;
    device::box_filter<WMAX><<<blocks, threads>>>(row->cbegin().get() + x,
						  _buf[x].begin().get(),
						  _rowWinSize,
						  strideI, strideB);
  // 左下
    auto	y = blocks.y*threads.y;
    std::advance(row, y);
    threads.x = BlockDim;
    blocks.x  = ncols/threads.x;
    threads.y = nrows%threads.y;
    blocks.y  = 1;
    device::box_filter<WMAX><<<blocks, threads>>>(row->cbegin().get(),
						  _buf[0].begin().get() + y,
						  _rowWinSize,
						  strideI, strideB);
  // 右下
    threads.x = ncols%threads.x;
    blocks.x  = 1;
    device::box_filter<WMAX><<<blocks, threads>>>(row->cbegin().get() + x,
						  _buf[x].begin().get() + y,
						  _rowWinSize,
						  strideI, strideB);

  // ---- 横方向積算 ----
    nrows -= (_rowWinSize - 1);
  // 左上
    threads.x = BlockDim;
    threads.y = BlockDim;
    blocks.x  = nrows/threads.x;
    blocks.y  = ncols/threads.y;
    device::box_filter<WMAX><<<blocks, threads>>>(_buf[0].cbegin().get(),
						  rowO->begin().get(),
						  _colWinSize,
						  strideB, strideO);
  // 左下
    y	      = blocks.y*threads.y;
    threads.y = ncols%threads.y;
    blocks.y  = 1;
    device::box_filter<WMAX><<<blocks, threads>>>(_buf[y].cbegin().get(),
						  rowO->begin().get() + y,
						  _colWinSize,
						  strideB, strideO);
  // 右上
    x	      = blocks.x*threads.x;
    std::advance(rowO, x);
    threads.x = nrows%threads.x;
    blocks.x  = 1;
    threads.y = BlockDim;
    blocks.y  = ncols/threads.y;
    device::box_filter<WMAX><<<blocks, threads>>>(_buf[0].cbegin().get() + x,
						  rowO->begin().get(),
						  _colWinSize,
						  strideB, strideO);
  // 右下
    threads.y = ncols%threads.y;
    blocks.y  = 1;
    device::box_filter<WMAX><<<blocks, threads>>>(_buf[y].cbegin().get() + x,
						  rowO->begin().get() + y,
						  _colWinSize,
						  strideB, strideO);
}

template <class T, size_t WMAX>
template <class ROW, class ROW_O, class OP> void
BoxFilter2<T, WMAX>::convolve(ROW rowL, ROW rowLe, ROW rowR, ROW_O rowO,
			      OP op, size_t disparitySearchWidth) const
{
    typedef typename std::iterator_traits<ROW_O>::value_type
						::value_type	value_type;
    
    auto	nrows = std::distance(rowL, rowLe);
    if (nrows < _rowWinSize)
	return;
    
    auto	ncols = std::distance(rowL->cbegin(), rowL->cend());
    if (ncols < _colWinSize)
	return;

    nrows -= (_rowWinSize - 1);
    
    _buf3.resize(nrows, ncols, disparitySearchWidth);
    
    const auto	strideL  = stride(rowL);
    const auto	strideR  = stride(rowR);
    const auto	strideD  = _buf3.stride();
    const auto	strideXD = ncols * strideD;

  // ---- 縦方向積算 ----
  // 視差左半かつ画像左半
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(disparitySearchWidth/threads.x, ncols/threads.y);
    device::box_filterV<WMAX><<<blocks, threads>>>(rowL->cbegin().get(),
						   rowR->cbegin().get(),
						   nrows,
						   _buf3[0][0].begin().get(),
						   op, _rowWinSize,
						   strideL, strideR,
						   strideD, strideXD);
  // 視差左半かつ画像右半
    const auto	x = blocks.y*threads.y;
    threads.y = ncols%threads.y;
    blocks.y  = 1;
    device::box_filterV<WMAX><<<blocks, threads>>>(
					rowL->cbegin().get() + x,
					rowR->cbegin().get() + x,
					nrows,
					_buf3[0][x].begin().get(),
					op, _rowWinSize,
					strideL, strideR, strideD, strideXD);
  // 視差右半かつ画像左半
    const auto	d = blocks.x*threads.x;
    threads.x = disparitySearchWidth%threads.x;
    blocks.x  = 1;
    threads.y = BlockDimY;
    blocks.y  = ncols/threads.y;
    device::box_filterV<WMAX><<<blocks, threads>>>(
					rowL->cbegin().get(),
					rowR->cbegin().get() + d,
					nrows,
					_buf3[0][0].begin().get() + d,
					op, _rowWinSize,
					strideL, strideR, strideD, strideXD);
  // 視差右半かつ画像右半
    threads.y = ncols%threads.y;
    blocks.y  = 1;
    device::box_filterV<WMAX><<<blocks, threads>>>(
					rowL->cbegin().get() + x,
					rowR->cbegin().get() + x + d,
					nrows,
					_buf3[0][x].begin().get() + d,
					op, _rowWinSize,
					strideL, strideR, strideD, strideXD);

  // ---- 横方向積算 ----
  // 視差左半かつ画像上半
    threads.x = BlockDimX;
    threads.y = BlockDimY;
    blocks.x = disparitySearchWidth/threads.x;
    blocks.y = nrows/threads.y;
    device::box_filterH<WMAX><<<blocks, threads>>>(
					_buf3[0][0].cbegin().get(),
					ncols,
					rowO->begin()->begin().get(),
					_colWinSize, strideXD, strideD);
  // 視差左半かつ画像下半
    const auto	y = blocks.y*threads.y;
    threads.y = nrows%threads.y;
    blocks.y  = 1;
    device::box_filterH<WMAX><<<blocks, threads>>>(
					_buf3[y][0].cbegin().get(),
					ncols,
					(rowO + y)->begin()->begin().get(),
					_colWinSize, strideXD, strideD);
  // 視差右半かつ画像上半
    threads.x = disparitySearchWidth%threads.x;
    blocks.x  = 1;
    threads.y = BlockDimY;
    blocks.y  = nrows/threads.y;
    device::box_filterH<WMAX><<<blocks, threads>>>(
					_buf3[0][0].cbegin().get() + d,
					ncols,
					rowO->begin()->begin().get() + d,
					_colWinSize, strideXD, strideD);
  // 視差右半かつ画像下半
    threads.y = nrows%threads.y;
    blocks.y  = 1;
    device::box_filterH<WMAX><<<blocks, threads>>>(
					_buf3[y][0].cbegin().get() + d,
					ncols,
					(rowO + y)->begin()->begin().get() + d,
					_colWinSize, strideXD, strideD);
}
#endif	// __NVCC__
}	// namespace cuda
}	// namespace TU
#endif	// !TU_CUDA_BOXFILTER_H
