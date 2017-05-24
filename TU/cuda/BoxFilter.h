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
	:_rowWinSize(rowWinSize), _colWinSize(colWinSize)		{}

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
    static constexpr size_t	BlockDimX  = 16;
    static constexpr size_t	BlockDimY  = 16;
    static constexpr size_t	BlockDim   = 128;
    
  private:
    size_t		_rowWinSize;
    size_t		_colWinSize;
    mutable Array2<T>	_buf;
};

#if defined(__NVCC__)
namespace device
{
/************************************************************************
*  __global__ functions							*
************************************************************************/
template <size_t WMAX, class COL, class COL_O> static __global__ void
box_filter(COL col, COL_O colO,
	   int rowWinSize, int colWinSize, int strideI, int strideO)
{
    typedef typename std::iterator_traits<COL>::value_type	in_type;
    typedef typename std::iterator_traits<COL_O>::value_type	out_type;

    constexpr auto	BlockDimX = BoxFilter2<out_type, WMAX>::BlockDimX;
    constexpr auto	BlockDimY = BoxFilter2<out_type, WMAX>::BlockDimY;
    __shared__ in_type	in_s[BlockDimY + WMAX][BlockDimX + WMAX];
    __shared__ out_type	mid_s[BlockDimY + WMAX][BlockDimX];

    const auto	u = blockIdx.x*blockDim.x + threadIdx.x;
    const auto	v = blockIdx.y*blockDim.y + threadIdx.y;
    col += (v*strideI + u);
    
    in_s[threadIdx.y][threadIdx.x] = *col;
    if (threadIdx.y < rowWinSize - 1)
    {
	in_s[blockDim.y + threadIdx.y][threadIdx.x]
	    = *(col + blockDim.y*strideI);
	if (threadIdx.x < colWinSize - 1)
	    in_s[blockDim.y + threadIdx.y][blockDim.x + threadIdx.x]
		= *(col + blockDim.y*strideI + blockDim.x);
    }
    if (threadIdx.x < colWinSize - 1)
	in_s[threadIdx.y][blockDim.x + threadIdx.x] = *(col + blockDim.x);

    __syncthreads();

    if (threadIdx.x == 0)	// 横方向に積算
    {
	const auto	p   = in_s[threadIdx.y];
	out_type	val = 0;
	for (int x = 0; x != colWinSize; ++x)
	    val += p[x];

	for (int x = 0; x != blockDim.x; ++x)
	{
	    mid_s[threadIdx.y][x] = val;
	    val += (p[x + colWinSize] - p[x]);
	}

	if (threadIdx.y < rowWinSize - 1)
	{
	    const auto	p   = in_s[blockDim.y + threadIdx.y];
	    out_type	val = 0;
	    for (int x = 0; x != colWinSize; ++x)
		val += p[x];

	    for (int x = 0; x != blockDim.x; ++x)
	    {
		mid_s[blockDim.y + threadIdx.y][x] = val;
		val += (p[x + colWinSize] - p[x]);
	    }
	}
    }

    __syncthreads();

    if (threadIdx.y == 0)	// 縦方向に積算
    {
	out_type	val = 0;
	for (int y = 0; y != rowWinSize; ++y)
	    val += mid_s[y][threadIdx.x];

	colO += (v*strideO + u);

	for (int y = 0; y != blockDim.y; ++y)
	{
	    *colO = val;
	    val += (mid_s[y + rowWinSize][threadIdx.x] - mid_s[y][threadIdx.x]);
	    colO += strideO;
	}
    }
}

template <size_t WMAX, class COL, class COL_O, class OP> static __global__ void
box_filterV(COL colL, COL colR, int nrow, COL_O colO, OP op, int rowWinSize,
	    int disparitySearchWidth, int strideL, int strideR, int strideO)
{
    typedef typename std::iterator_traits<COL>::value_type	in_type;
    typedef typename std::iterator_traits<COL_O>::value_type	out_type;

    constexpr auto	BlockDimX = BoxFilter2<out_type, WMAX>::BlockDimX;
    constexpr auto	BlockDimY = BoxFilter2<out_type, WMAX>::BlockDimY;
    __shared__ in_type	inL[WMAX][BlockDimY + 1];
    __shared__ in_type	inR[WMAX][BlockDimY + BlockDimX + 1];

    const auto	d = blockIdx.x*blockDim.x + threadIdx.x;
    const auto	x = blockIdx.y*blockDim.y + threadIdx.y;
    colL += x;
    colR += (x + d);
    
    out_type	val = 0;
    for (int i = 0; i != rowWinSize - 1; ++i)
    {
	val += op(inL[i][threadIdx.y] = *colL,
		  inR[i][threadIdx.y + threadIdx.x] = *colR);
	colL += strideL;
	colR += strideR;
    }

    colO += (x*disparitySearchWidth + d);
    for (int i = rowWinSize - 1; nrow != 0; --nrow)
    {
	*colO = (val += op(inL[i][threadIdx.y] = *colL,
			   inR[i][threadIdx.y + threadIdx.x] = *colR));
	colL += strideL;
	colR += strideR;
	colO += strideO;

	if (++i == rowWinSize)
	    i = 0;
	val -= op(inL[i][threadIdx.y], inR[i][threadIdx.y + threadIdx.x]);
    }
}

template <size_t WMAX, class COL, class COL_O> static __global__ void
box_filterH(COL col, int ncol, COL_O colO,
	    int colWinSize, int disparitySearchWidth, int strideI, int strideO)
{
    typedef typename std::iterator_traits<COL_O>::value_type	out_type;

    constexpr auto	BlockDim = BoxFilter2<out_type, WMAX>::BlockDim;
    __shared__ out_type	in_s[WMAX][BlockDim];

    const auto	dy = blockIdx.x*blockDim.x + threadIdx.x;
    const auto	y  = dy/disparitySearchWidth;
    const auto	d  = dy%disparitySearchWidth;

    col += (y*strideI + d);
    
    out_type	val = 0;
    for (int i = 0; i != colWinSize - 1; ++i)
    {
	val += (in_s[i][threadIdx.x] = *col);
	col += disparitySearchWidth;
    }

    colO += (y*strideO + d);

    for (int i = colWinSize - 1; ncol != 0; --ncol)
    {
	*colO = (val += (in_s[i][threadIdx.x] = *col));
	col  += disparitySearchWidth;
	colO += disparitySearchWidth;

	if (++i == colWinSize)
	    i = 0;
	val -= in_s[i][threadIdx.x];
    }
}

}	// namespace device

/************************************************************************
*  class BoxFilter2<T, WMAX>						*
************************************************************************/
template <class T, size_t WMAX> template <class ROW, class ROW_O> void
BoxFilter2<T, WMAX>::convolve(ROW row, ROW rowe, ROW_O rowO) const
{
    auto	nrow = std::distance(row, rowe);
    if (nrow < _rowWinSize)
	return;

    auto	ncol = std::distance(row->cbegin(), row->cend());
    if (ncol < _colWinSize)
	return;

    nrow -= (_rowWinSize - 1);
    ncol -= (_colWinSize - 1);

    const auto	strideI = stride(row);
    const auto	strideO = stride(rowO);

  // 左上
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(ncol/threads.x, nrow/threads.y);
    device::box_filter<WMAX><<<blocks, threads>>>(row->cbegin(), rowO->begin(),
						  _rowWinSize, _colWinSize,
						  strideI, strideO);
  // 右上
    const auto	x = blocks.x*threads.x;
    threads.x = ncol - x;
    blocks.x  = 1;
    device::box_filter<WMAX><<<blocks, threads>>>(row->cbegin() + x,
						  rowO->begin() + x,
						  _rowWinSize, _colWinSize,
						  strideI, strideO);
  // 左下
    const auto	y = blocks.y*threads.y;
    std::advance(row,  y);
    std::advance(rowO, y);
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    threads.y = nrow - y;
    blocks.y  = 1;
    device::box_filter<WMAX><<<blocks, threads>>>(row->cbegin(), rowO->begin(),
						  _rowWinSize, _colWinSize,
						  strideI, strideO);
  // 右下
    threads.x = ncol - x;
    blocks.x  = 1;
    device::box_filter<WMAX><<<blocks, threads>>>(row->cbegin() + x,
						  rowO->begin() + x,
						  _rowWinSize, _colWinSize,
						  strideI, strideO);
}

template <class T, size_t WMAX> template <class ROW, class ROW_O, class OP> void
BoxFilter2<T, WMAX>::convolve(ROW rowL, ROW rowLe, ROW rowR, ROW_O rowO,
			      OP op, size_t disparitySearchWidth) const
{
    typedef typename std::iterator_traits<ROW_O>::value_type
	::value_type	value_type;

    auto	nrow = std::distance(rowL, rowLe);		    // 行数
    if (nrow < _rowWinSize)
	return;

    nrow -= (_rowWinSize - 1);

    auto	ncol = std::distance(rowL->cbegin(), rowL->cend()); // 列数
    if (ncol < _colWinSize)
	return;

    _buf.resize(nrow, disparitySearchWidth*ncol);
    
    const auto	strideL = stride(rowL);
    const auto	strideR = stride(rowR);
    const auto	strideB = _buf.stride();
    const auto	strideO = stride(rowO);

  // 縦方向畳み込み：視差左半かつ画像左半
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(disparitySearchWidth/threads.x, ncol/threads.y);
    device::box_filterV<WMAX><<<blocks, threads>>>(
	rowL->cbegin(), rowR->cbegin(), nrow, _buf[0].begin(),
	op, _rowWinSize, disparitySearchWidth, strideL, strideR, strideB);

  // 縦方向畳み込み：視差右半かつ画像左半
    auto	d = blocks.x*threads.x;
    threads.x = disparitySearchWidth%threads.x;
    blocks.x  = 1;
    device::box_filterV<WMAX><<<blocks, threads>>>(
	rowL->cbegin(), rowR->cbegin() + d, nrow, _buf[0].begin() + d,
	op, _rowWinSize, disparitySearchWidth, strideL, strideR, strideB);

  // 縦方向畳み込み：視差左半かつ画像右半
    const auto	x = blocks.y*threads.y;
    threads.x = BlockDimX;
    blocks.x  = disparitySearchWidth/threads.x;
    threads.y = ncol%threads.y;
    blocks.y  = 1;
    device::box_filterV<WMAX><<<blocks, threads>>>(
	rowL->cbegin() + x, rowR->cbegin() + x, nrow,
	_buf[0].begin() + x*disparitySearchWidth,
	op, _rowWinSize, disparitySearchWidth, strideL, strideR, strideB);

  // 縦方向畳み込み：視差右半かつ画像右半
    threads.x = disparitySearchWidth%threads.x;
    blocks.x  = 1;
    device::box_filterV<WMAX><<<blocks, threads>>>(
	rowL->cbegin() + x, rowR->cbegin() + x + d, nrow,
	_buf[0].begin() + x*disparitySearchWidth + d,
	op, _rowWinSize, disparitySearchWidth, strideL, strideR, strideB);

  // 横方向畳み込み：前半
    ncol -= (_colWinSize - 1);
    threads.x = BlockDim;
    blocks.x  = (disparitySearchWidth*nrow)/threads.x;
    threads.y = 1;
    blocks.y  = 1;
    device::box_filterH<WMAX><<<blocks, threads>>>(
	_buf[0].cbegin(), ncol, rowO->begin(),
	_colWinSize, disparitySearchWidth, strideB, strideO);
    
  // 横方向畳み込み：後半
    const auto	dy = blocks.x*threads.x;
    const auto	y  = dy/disparitySearchWidth;
    d = dy%disparitySearchWidth;
    std::advance(rowO, y);
    threads.x = (disparitySearchWidth*nrow)%threads.x;
    blocks.x  = 1;
    device::box_filterH<WMAX><<<blocks, threads>>>(
	_buf[y].cbegin() + d, ncol, rowO->begin() + d,
	_colWinSize, disparitySearchWidth, strideB, strideO);
}
#endif	// __NVCC__
}	// namespace cuda
}	// namespace TU
#endif	// !TU_CUDA_BOXFILTER_H
