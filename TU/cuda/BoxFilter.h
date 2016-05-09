/*
 *  $Id: BoxFilter.h 1962 2016-03-22 02:56:32Z ueshiba $
 */
/*!
  \file		BoxFilter.h
  \brief	boxフィルタの定義と実装
*/ 
#ifndef __TU_CUDA_BOXFILTER_H
#define __TU_CUDA_BOXFILTER_H

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
box_filter(COL col, COL cole, COL_O colO, int winSize, int strideI, int strideO)
{
    typedef typename std::iterator_traits<COL_O>::value_type	out_type;

    const auto	y = blockIdx.x*blockDim.x + threadIdx.x;
    col  += y*strideI;
    cole += y*strideI;
    colO += y;

    __shared__ out_type	val_s[BlockDimX][WMAX];
    const auto		p = val_s[threadIdx.x];
    auto		val = 0;
    for (int i = 0; i != winSize; ++i, ++col)
	val += (p[i] = *col);
    *colO = val;

    for (int i = 0; col != cole; ++i, ++col)
    {
	if (i == winSize)
	    i = 0;
	val -= p[i];
	*(colO += strideO) = (val += (p[i] = *col));
    }
}

template <size_t WMAX, class COL, class COL_O, class OP> static __global__ void
box_filter(COL colL, COL colLe, COL colR, COL_O colO, OP op,
	   int winSize, int disparitySearchWidth,
	   int strideL, int strideR, int strideO)
{
    typedef typename std::iterator_traits<COL>::value_type	in_type;
    typedef typename std::iterator_traits<COL_O>::value_type	out_type;

    const auto	d = blockIdx.x*blockDim.x + threadIdx.x;	// 視差
    const auto	y = blockIdx.y*blockDim.y + threadIdx.y;	// 行
    const auto	ncol = colLe - colL;
    colL  += y*strideL;
    colLe += y*strideL;
    colR  += y*strideR + d;
    colO  += d*strideO + y;

    __shared__ in_type	inL[BlockDimY][WMAX];
    __shared__ in_type	inR[BlockDimY][BlockDimX + WMAX];

    const auto	p = inL[threadIdx.y];
    const auto	q = inR[threadIdx.y];
    const int	t = threadIdx.x;
    int		b = blockDim.x;
    q[t] = *colR;
    colR += b;
    
  // 最初のwinSize画素分の相違度を計算してvalに積算
    out_type		val = 0;
    for (int i = 0; i != winSize; ++i, ++colL)
    {
	if (t == 0)
	{
	    p[i    ] = *colL;
	    q[i + b] = *colR;
	    ++colR;
	}
	__syncthreads();

	val += op(p[i], q[i + t]);
    }
    *colO = val;

  // 逐次的にvalを更新して出力
    strideO *= disparitySearchWidth;
    b += winSize;
    for (int i = 0, j = t, k = j + winSize;
	 colL != colLe; ++i, ++j, ++k, ++colL)
    {
	if (i == winSize)
	    i = 0;
	if (j == b)
	    j = 0;
	if (k == b)
	    k = 0;
	
	val -= op(p[i], q[j]);

	if (t == 0)
	{
	    p[i] = *colL;
	    q[j] = *colR;
	    ++colR;
	}
	__syncthreads();

	*(colO += strideO) = (val += op(p[i], q[k]));
    }
}
}	// namespace device
/************************************************************************
*  class BoxFilter2							*
************************************************************************/
template <class T, size_t WMAX> template <class ROW, class ROW_O> void
BoxFilter2<T, WMAX>::convolve(ROW row, ROW rowe, ROW_O rowO) const
{
    const auto	nrow = std::distance(row, rowe);
    if (nrow < _rowWinSize)
	return;
    
    const auto	ncol = std::distance(row->cbegin(), row->cend());
    if (ncol < _colWinSize)
	return;
    
    _buf.resize(ncol, nrow);
    
    const auto	strideI = stride(row);
    const auto	strideB = _buf.stride();
    const auto	strideO = stride(rowO);

  // 上
    dim3	threads(BlockDimX);
    dim3	blocks(nrow/threads.x);
    device::box_filter<WMAX><<<blocks, threads>>>(row->cbegin(), row->cend(),
						  _buf.begin()->begin(),
						  _colWinSize,
						  strideI, strideB);
  // 下
    auto	y = blocks.x*threads.x;
    std::advance(row, y);
    if (row != rowe)
    {
	threads.x = nrow%threads.x;
	blocks.x  = 1;
	device::box_filter<WMAX><<<blocks, threads>>>(row->cbegin(),
						      row->cend(),
						      _buf.begin()->begin() + y,
						      _colWinSize,
						      strideI, strideB);
    }
  // 上
    auto	rowB = _buf.cbegin();
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    device::box_filter<WMAX><<<blocks, threads>>>(rowB->cbegin(), rowB->cend(),
						  rowO->begin(),
						  _rowWinSize,
						  strideB, strideO);
  // 下
    y = blocks.x*threads.x;
    std::advance(rowB, y);
    if (rowB != _buf.cend())
    {
	threads.x = ncol%threads.x;
	blocks.x  = 1;
	device::box_filter<WMAX><<<blocks, threads>>>(rowB->cbegin(),
						      rowB->cend(),
						      rowO->begin() + y,
						      _rowWinSize,
						      strideB, strideO);
    }
}

template <class T, size_t WMAX> template <class ROW, class ROW_O, class OP> void
BoxFilter2<T, WMAX>::convolve(ROW rowL, ROW rowLe, ROW rowR, ROW_O rowO,
			      OP op, size_t disparitySearchWidth) const
{
    typedef typename std::iterator_traits<ROW_O>::value_type
						::value_type	value_type;
    
    const auto	nrow = std::distance(rowL, rowLe);		    // 行数
    if (nrow < _rowWinSize)
	return;
    
    const auto	ncol = std::distance(rowL->cbegin(), rowL->cend()); // 列数
    if (ncol < _colWinSize)
	return;

    _buf.resize(ncol*disparitySearchWidth, nrow);
    
    const auto	strideL = stride(rowL);
    const auto	strideR = stride(rowR);
    const auto	strideB = _buf.stride();
    const auto	strideO = stride(rowO);

  // 画像上半かつ視差左半
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(disparitySearchWidth/threads.x, nrow/threads.y);
    device::box_filter<WMAX><<<blocks, threads>>>(rowL->cbegin(), rowL->cend(),
						  rowR->cbegin(),
						  _buf[0].begin(), op, 
						  _colWinSize,
						  disparitySearchWidth,
						  strideL, strideR, strideB);
  // 画像上半かつ視差右半
    auto	d = blocks.x*threads.x;
    threads.x = disparitySearchWidth%threads.x;
    blocks.x  = 1;
    device::box_filter<WMAX><<<blocks, threads>>>(rowL->cbegin(), rowL->cend(),
						  rowR->cbegin() + d,
						  _buf[d].begin(), op,
						  _colWinSize,
						  disparitySearchWidth,
						  strideL, strideR, strideB);
  // 画像下半
    std::advance(rowL, blocks.y*threads.y);
    if (rowL != rowLe)
    {
      // 画像下半かつ視差左半
	std::advance(rowR, blocks.y*threads.y);
	threads.x = BlockDimX;
	blocks.x  = disparitySearchWidth/threads.x;
	threads.y = nrow%threads.y;
	blocks.y  = 1;
	device::box_filter<WMAX><<<blocks, threads>>>(rowL->cbegin(),
						      rowL->cend(),
						      rowR->cbegin(),
						      _buf[0].begin(), op,
						      _colWinSize,
						      disparitySearchWidth,
						      strideL, strideR,
						      strideB);
      // 画像上半かつ視差右半
	threads.x = disparitySearchWidth%threads.x;
	blocks.x  = 1;
	device::box_filter<WMAX><<<blocks, threads>>>(rowL->cbegin(),
						      rowL->cend(),
						      rowR->cbegin() + d,
						      _buf[d].begin(), op,
						      _colWinSize,
						      disparitySearchWidth,
						      strideL, strideR,
						      strideB);
    }

  // 左半
    auto	rowB = _buf.cbegin();
    threads.x = BlockDimX;
    blocks.x  = _buf.nrow()/threads.x;
    threads.y = 1;
    blocks.y  = 1;
    device::box_filter<WMAX><<<blocks, threads>>>(rowB->cbegin(), rowB->cend(),
						  rowO->begin(), _rowWinSize,
						  strideB, strideO);
  // 右半
    d = blocks.x*threads.x;
    std::advance(rowB, d);
    if (rowB != _buf.cend())
    {
	threads.x = _buf.nrow()%threads.x;
	blocks.x  = 1;
	device::box_filter<WMAX><<<blocks, threads>>>(rowB->cbegin(),
						      rowB->cend(),
						      rowO->begin() + d,
						      _rowWinSize,
						      strideB, strideO);
    }
}
#endif	// __NVCC__
}	// namespace cuda
}	// namespace TU
#endif	// !__TU_CUDA_BOXFILTER_H
