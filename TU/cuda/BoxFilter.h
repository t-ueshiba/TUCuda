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
*  class BoxFilter2							*
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

  private:
    size_t		_rowWinSize;
    size_t		_colWinSize;
    mutable Array2<T>	_buf;
};

#if defined(__NVCC__)
/************************************************************************
*  __global__ functions							*
************************************************************************/
template <size_t WMAX, class COL, class COL_O> static __global__ void
box_filter_kernel(COL col, COL cole, COL_O colO,
		  int winSize, int stride_i, int stride_o)
{
    typedef typename std::iterator_traits<COL_O>::value_type	out_type;

    const auto	y = blockIdx.x*blockDim.x + threadIdx.x;
    col  += y*stride_i;
    cole += y*stride_i;
    colO += y;

    __shared__ out_type	val_s[BlockDimX][WMAX];
    const auto		p = val_s[threadIdx.x];
    auto		val = (p[0] = *col);
    int			i = 0;
    
    while (++i != winSize)
	val += (p[i] = *++col);
    *colO = val;

    while (++col != cole)
    {
	if (i == winSize)
	    i = 0;
	val -= p[i];
	*(colO += stride_o) = (val += (p[i++] = *col));
    }
}

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
    
    const auto	stride_i = stride(row);
    const auto	stride_o = stride(rowO);

    _buf.resize(ncol, nrow);
    
  // 上
    const auto	stride_b = stride(_buf.cbegin());
    dim3	threads(BlockDimX);
    dim3	blocks(nrow/threads.x);
    box_filter_kernel<WMAX><<<blocks, threads>>>(row->cbegin(), row->cend(),
						 _buf.begin()->begin(),
						 _colWinSize,
						 stride_i, stride_b);
  // 下
    auto	y = blocks.x*threads.x;
    std::advance(row, y);
    if (row != rowe)
    {
	threads.x = nrow%threads.x;
	blocks.x  = 1;
	box_filter_kernel<WMAX><<<blocks, threads>>>(row->cbegin(), row->cend(),
						     _buf.begin()->begin() + y,
						     _colWinSize,
						     stride_i, stride_b);
    }

  // 上
    auto	rowb = _buf.cbegin();
    threads.x = BlockDimX;
    blocks.x  = ncol/threads.x;
    box_filter_kernel<WMAX><<<blocks, threads>>>(rowb->cbegin(), rowb->cend(),
						 rowO->begin(),
						 _rowWinSize,
						 stride_b, stride_o);
  // 下
    y = blocks.x*threads.x;
    std::advance(rowb, y);
    if (rowb != _buf.cend())
    {
	threads.x = ncol%threads.x;
	blocks.x  = 1;
	box_filter_kernel<WMAX><<<blocks, threads>>>(rowb->cbegin(),
						     rowb->cend(),
						     rowO->begin() + y,
						     _rowWinSize,
						     stride_b, stride_o);
    }
}
#endif	// __NVCC__
}
}
#endif	// !__TU_CUDA_BOXFILTER_H
