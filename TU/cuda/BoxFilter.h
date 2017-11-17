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
template <class FILTER, class COL, class COL_O> __global__ void
box_filter(COL col, COL_O colO, int winSize, int strideI, int strideO)
{
    using value_type  =	typename FILTER::value_type;
    
    constexpr auto	BlockDim   = FILTER::BlockDim;
    constexpr auto	WinSizeMax = FILTER::WinSizeMax;
    
    __shared__ value_type	in_s[BlockDim + WinSizeMax - 1][BlockDim + 1];
    __shared__ value_type	out_s[BlockDim][BlockDim + 1];

    const auto	x0 = __mul24(blockIdx.x, blockDim.x);	// ブロック左上隅
    const auto	y0 = __mul24(blockIdx.y, blockDim.y);	// ブロック左上隅

    loadTileV(col + __mul24(y0, strideI) + x0, strideI, in_s, winSize - 1);
    __syncthreads();
    
    if (threadIdx.y == 0)
    {
      // 各列を並列に縦方向積算
	out_s[0][threadIdx.x] = in_s[0][threadIdx.x];
	for (int y = 1; y != winSize; ++y)
	    out_s[0][threadIdx.x] += in_s[y][threadIdx.x];

	for (int y = 1; y != blockDim.y; ++y)
	    out_s[y][threadIdx.x]
		= out_s[y-1][threadIdx.x]
		+ in_s[y-1+winSize][threadIdx.x] - in_s[y-1][threadIdx.x];
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

template <class FILTER, class COL, class COL_O, class OP> __global__ void
box_filterV(COL colL, COL colR, int nrows, COL_O colO, OP op, int rowWinSize,
	    int strideL, int strideR, int strideD, int strideXD)
{
    using value_type  =	typename FILTER::value_type;
    
    constexpr auto	BlockDimX  = FILTER::BlockDimX;
    constexpr auto	BlockDimY  = FILTER::BlockDimY;
    constexpr auto	WinSizeMax = FILTER::WinSizeMax;

    __shared__ value_type	val_s[WinSizeMax][BlockDimY][BlockDimX + 1];

    const auto	d = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// 視差
    const auto	x = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;	// 列

    colL += x;
    colR += (x + d);
    colO += (__mul24(x, strideD) + d);

  // スレッド間で重複せずに左右の画素をロードするために threadIdx.y == 0
  // なるスレッドのみが読み込むようにすると，読み込む度に__syncthreads()
  // が必要になり，かえって性能が上がらない．何も工夫せずに colL と colR
  // を直接 dereference するのが最も高速．(2017.11.11)

  // 最初のwinSize画素分の相違度を計算してvalに積算
    auto	val = val_s[0][threadIdx.y][threadIdx.x] = op(*colL, *colR);
    for (int i = 0; ++i != rowWinSize; )
	val += (val_s[i][threadIdx.y][threadIdx.x] = op(*(colL += strideL),
							*(colR += strideR)));
    *colO = val;
    
  // 逐次的にvalを更新して出力
    for (int i = 0; --nrows; )
    {
	val -= val_s[i][threadIdx.y][threadIdx.x];
	*(colO += strideXD) = (val += (val_s[i][threadIdx.y][threadIdx.x]
				       = op(*(colL += strideL),
					    *(colR += strideR))));

	if (++i == rowWinSize)
	    i = 0;
    }
}

template <class FILTER, class COL> __global__ void
box_filterH(COL col, int ncols, int colWinSize, int strideXD, int strideD)
{
    using value_type  =	typename FILTER::value_type;
    
    constexpr auto	BlockDimX  = FILTER::BlockDimX;
    constexpr auto	BlockDimY  = FILTER::BlockDimY;
    constexpr auto	WinSizeMax = FILTER::WinSizeMax;

    __shared__ value_type	val_s[WinSizeMax][BlockDimY][BlockDimX + 1];

    const auto	d = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// 視差
    const auto	y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;	// 行

    col += (__mul24(y, strideXD) + d);

    auto	colO = col;
    auto	val = (val_s[0][threadIdx.y][threadIdx.x] = *col);
    for (int i = 0; ++i < colWinSize; )
	val += (val_s[i][threadIdx.y][threadIdx.x] = *(col += strideD));
    *colO = val;
    
    for (int i = 0; --ncols; )
    {
	val -= val_s[i][threadIdx.y][threadIdx.x];
	*(colO += strideD) = (val += (val_s[i][threadIdx.y][threadIdx.x]
				      = *(col += strideD)));

	if (++i == colWinSize)
	    i = 0;
    }
}
}	// namespace device
#endif	// __NVCC__
    
/************************************************************************
*  class BoxFilter2<T, WMAX>						*
************************************************************************/
//! CUDAによる2次元boxフィルタを表すクラス
template <class T, size_t WMAX=23>
class BoxFilter2
{
  public:
    using	value_type = T;
    
    constexpr static size_t	WinSizeMax = WMAX;
    constexpr static size_t	BlockDim   = 16;
    constexpr static size_t	BlockDimX  = 32;
    constexpr static size_t	BlockDimY  = 16;
    
  public:
  //! CUDAによる2次元boxフィルタを生成する．
  /*!
    \param rowWinSize	boxフィルタのウィンドウの行幅(高さ)
    \param colWinSize	boxフィルタのウィンドウの列幅(幅)
   */	
		BoxFilter2(size_t rowWinSize, size_t colWinSize)
		    :_rowWinSize(rowWinSize), _colWinSize(colWinSize)
		{
		    if (_rowWinSize > WinSizeMax || _colWinSize > WinSizeMax)
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
    
  //! 与えられた行幅(高さ)を持つ入力データ列に対する出力データ列の行幅を返す．
  /*!
    \param inRowLength	入力データ列の行幅
    \return		出力データ列の行幅
   */
    size_t	outRowLength(size_t inRowLength) const
		{
		    return inRowLength + 1 - rowWinSize();
		}
    
  //! 与えられた列幅(幅)を持つ入力データ列に対する出力データ列の列幅を返す．
  /*!
    \param inColLength	入力データ列の列幅
    \return		出力データ列の列幅
   */
    size_t	outColLength(size_t inColLength) const
		{
		    return inColLength + 1 - colWinSize();
		}

    size_t	overlap()	const	{ return rowWinSize() - 1; }

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

    template <class COL, class COL_O>
    void	convolve(COL col, COL_O colO,
			 size_t nrows,   size_t ncols,
			 size_t strideI, size_t strideO)	const	;
    
  private:
    size_t		_rowWinSize;
    size_t		_colWinSize;
    mutable Array2<T>	_buf;
};

#if defined(__NVCC__)
template <class T, size_t WMAX> template <class ROW, class ROW_O> void
BoxFilter2<T, WMAX>::convolve(ROW row, ROW rowe, ROW_O rowO) const
{
    const auto	nrows = std::distance(row, rowe);
    if (nrows < _rowWinSize)
	return;
    
    const auto	ncols = std::distance(std::cbegin(*row), std::cend(*row));
    if (ncols < _colWinSize)
	return;

    convolve(std::cbegin(*row), std::begin(*rowO),
	     nrows, ncols, stride(row), stride(rowO));
}

template <class T, size_t WMAX> template <class COL, class COL_O> void
BoxFilter2<T, WMAX>::convolve(COL col, COL_O colO,
			      size_t nrows, size_t ncols,
			      size_t strideI, size_t strideO) const
{
    ncols -= (_colWinSize - 1);

    _buf.resize(32, ncols, nrows);

    const auto	strideB = _buf.stride();

  // ---- 縦方向積算 ----
  // 左上
    dim3	threads(BlockDim, BlockDim);
    dim3	blocks(ncols/threads.x, nrows/threads.y);
    device::box_filter<BoxFilter2><<<blocks, threads>>>(
						col,
						std::begin(_buf[0]),
						_rowWinSize,
						strideI, strideB);
  // 右上
    const auto	x = blocks.x*threads.x;
    threads.x = ncols - x;
    blocks.x  = 1;
    device::box_filter<BoxFilter2><<<blocks, threads>>>(
						col + x,
						std::begin(_buf[x]),
						_rowWinSize,
						strideI, strideB);
  // 左下
    auto	y = blocks.y*threads.y;
    col += y*strideI;
    threads.x = BlockDim;
    blocks.x  = ncols/threads.x;
    threads.y = nrows - y;
    blocks.y  = 1;
    device::box_filter<BoxFilter2><<<blocks, threads>>>(
						col,
						std::begin(_buf[0]) + y,
						_rowWinSize,
						strideI, strideB);
  // 右下
    threads.x = ncols - x;
    blocks.x  = 1;
    device::box_filter<BoxFilter2><<<blocks, threads>>>(
						col + x,
						std::begin(_buf[x]) + y,
						_rowWinSize,
						strideI, strideB);

  // ---- 横方向積算 ----
    nrows -= (_rowWinSize - 1);
  // 左上
    threads.x = BlockDim;
    blocks.x  = nrows/threads.x;
    threads.y = BlockDim;
    blocks.y  = ncols/threads.y;
    device::box_filter<BoxFilter2><<<blocks, threads>>>(
						std::cbegin(_buf[0]),
						colO,
						_colWinSize,
						strideB, strideO);
  // 左下
    threads.y = ncols - x;
    blocks.y  = 1;
    device::box_filter<BoxFilter2><<<blocks, threads>>>(
						std::cbegin(_buf[x]),
						colO + x,
						_colWinSize,
						strideB, strideO);
  // 右上
    y	      = blocks.x*threads.x;
    colO += y*strideO;
    threads.x = nrows - y;
    blocks.x  = 1;
    threads.y = BlockDim;
    blocks.y  = ncols/threads.y;
    device::box_filter<BoxFilter2><<<blocks, threads>>>(
						std::cbegin(_buf[0]) + y,
						colO,
						_colWinSize,
						strideB, strideO);
  // 右下
    threads.y = ncols - x;
    blocks.y  = 1;
    device::box_filter<BoxFilter2><<<blocks, threads>>>(
						std::cbegin(_buf[x]) + y,
						colO + x,
						_colWinSize,
						strideB, strideO);
}

template <class T, size_t WMAX>
template <class ROW, class ROW_O, class OP> void
BoxFilter2<T, WMAX>::convolve(ROW rowL, ROW rowLe, ROW rowR, ROW_O rowO,
			      OP op, size_t disparitySearchWidth) const
{
    auto	nrows = std::distance(rowL, rowLe);
    if (nrows < _rowWinSize)
	return;
    
    auto	ncols = std::distance(std::cbegin(*rowL), std::cend(*rowL));
    if (ncols < _colWinSize)
	return;

    const auto	strideL  = stride(rowL);
    const auto	strideR  = stride(rowR);
    const auto	strideD  = stride(std::cbegin(*rowO));
    const auto	strideXD = ncols * strideD;

  // ---- 縦方向積算 ----
    nrows -= (_rowWinSize - 1);
  // 視差左半かつ画像左半
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(disparitySearchWidth/threads.x, ncols/threads.y);
    device::box_filterV<BoxFilter2><<<blocks, threads>>>(
				std::cbegin(*rowL),
				std::cbegin(*rowR),
				nrows,
				std::begin(*std::begin(*rowO)),
				op, _rowWinSize,
				strideL, strideR, strideD, strideXD);
  // 視差左半かつ画像右半
    const auto	x = blocks.y*threads.y;
    threads.y = ncols - x;
    blocks.y  = 1;
    device::box_filterV<BoxFilter2><<<blocks, threads>>>(
				std::cbegin(*rowL) + x,
				std::cbegin(*rowR) + x,
				nrows,
				std::begin(*(std::begin(*rowO) + x)),
				op, _rowWinSize,
				strideL, strideR, strideD, strideXD);
  // 視差右半かつ画像左半
    const auto	d = blocks.x*threads.x;
    threads.x = disparitySearchWidth - d;
    blocks.x  = 1;
    threads.y = BlockDimY;
    blocks.y  = ncols/threads.y;
    device::box_filterV<BoxFilter2><<<blocks, threads>>>(
				std::cbegin(*rowL),
				std::cbegin(*rowR) + d,
				nrows,
				std::begin(*std::begin(*rowO)) + d,
				op, _rowWinSize,
				strideL, strideR, strideD, strideXD);
  // 視差右半かつ画像右半
    threads.y = ncols - x;
    blocks.y  = 1;
    device::box_filterV<BoxFilter2><<<blocks, threads>>>(
				std::cbegin(*rowL) + x,
				std::cbegin(*rowR) + x + d,
				nrows,
				std::begin(*(std::begin(*rowO) + x)) + d,
				op, _rowWinSize,
				strideL, strideR, strideD, strideXD);

  // ---- 横方向積算 ----
  // 視差左半かつ画像上半
    threads.x = BlockDimX;
    threads.y = BlockDimY;
    blocks.x = disparitySearchWidth/threads.x;
    blocks.y = nrows/threads.y;
    device::box_filterH<BoxFilter2><<<blocks, threads>>>(
				std::begin(*std::begin(*rowO)),
				ncols, _colWinSize, strideXD, strideD);
  // 視差左半かつ画像下半
    const auto	y = blocks.y*threads.y;
    threads.y = nrows - y;
    blocks.y  = 1;
    device::box_filterH<BoxFilter2><<<blocks, threads>>>(
				std::begin(*std::begin(*(rowO + y))),
				ncols, _colWinSize, strideXD, strideD);
  // 視差右半かつ画像上半
    threads.x = disparitySearchWidth%threads.x;
    blocks.x  = 1;
    threads.y = BlockDimY;
    blocks.y  = nrows/threads.y;
    device::box_filterH<BoxFilter2><<<blocks, threads>>>(
				std::begin(*std::begin(*rowO))+ d,
				ncols, _colWinSize, strideXD, strideD);
  // 視差右半かつ画像下半
    threads.y = nrows - y;
    blocks.y  = 1;
    device::box_filterH<BoxFilter2><<<blocks, threads>>>(
				std::begin(*std::begin(*(rowO + y))) + d,
				ncols, _colWinSize, strideXD, strideD);
}
#endif	// __NVCC__
}	// namespace cuda
}	// namespace TU
#endif	// !TU_CUDA_BOXFILTER_H
