/*
 *  $Id$
 */
/*!
  \mainpage	libTUCuda++ - NVIDIA社のCUDAを利用するためのユティリティライブラリ
  \anchor	libTUCuda

  \section copyright 著作権
  平成14-23年（独）産業技術総合研究所 著作権所有

  創作者：植芝俊夫

  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
  等の行為を禁止します．
   
  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
  よび創作者は責任を負いません。

  Copyright 2002-2011.
  National Institute of Advanced Industrial Science and Technology (AIST)

  Creator: Toshio UESHIBA

  [AIST Confidential and all rights reserved.]
  This program is confidential. Any using, copying, changing or
  giving any information concerning with this program to others
  without permission by the copyright holder are strictly prohibited.

  [No Warranty.]
  The copyright holder or the creator are not responsible for any
  damages caused by using this program.

  \section abstract 概要
  libTUCuda++は，C++環境においてNVIDIA社のCUDAを利用するためのユティリティ
  ライブラリである．以下のようなクラスおよび関数が実装されている．

  <b>デバイス側のグローバルメモリを確保するアロケータ
  - #TU::cuda::allocator
  
  <b>デバイス側のグローバルメモリ領域にマップされるホスト側メモリを確保するアロケータ
  - #TU::cuda::mapped_allocator
  
  <b>デバイス側のグローバルメモリ領域にとられる1次元および2次元配列</b>
  - #TU::cuda::Array
  - #TU::cuda::Array2

  <b>デバイス側のグローバルメモリ領域にマップされるホスト側1次元および2次元配列</b>
  - #TU::cuda::MappedArray
  - #TU::cuda::MappedArray2

  <b>デバイス側のテクスチャメモリ</b>
  - #TU::cuda::Texture
  
  <b>フィルタリング</b>
  - #TU::cuda::FIRFilter2
  - #TU::cuda::FIRGaussianConvolver2
  - #TU::cuda::BoxFilter2

  <b>アルゴリズム</b>
  - #TU::cuda::copyToConstantMemory(ITER, ITER, T*)
  - #TU::cuda::subsample(IN, IN, OUT)
  - #TU::cuda::op3x3(IN, IN, OUT, OP)
  - #TU::cuda::suppressNonExtrema3x3(IN, IN, OUT, OP, typename std::iterator_traits<IN>::value_type::value_type)
  
  <b>時間計測</b>
  - #TU::cuda::clock

  \file		Array++.h
  \brief	CUDAデバイス上の配列に関連するクラスの定義と実装
*/
#ifndef TU_CUDA_ARRAYPP_H
#define TU_CUDA_ARRAYPP_H

#include <thrust/copy.h>
#include <thrust/fill.h>
#include "TU/cuda/allocator.h"
#include "TU/Array++.h"

//! 植芝によって開発されたクラスおよび関数を納める名前空間
namespace TU
{
/************************************************************************
*  specialization for BufTraits<T, ALLOC> for CUDA			*
************************************************************************/
template <class T>
class BufTraits<T, cuda::allocator<T> >
    : public std::allocator_traits<cuda::allocator<T> >
{
  private:
    using super			= std::allocator_traits<cuda::allocator<T> >;

  public:
    using iterator		= typename super::pointer;
    using const_iterator	= typename super::const_pointer;
    
  protected:
    using pointer		= typename super::pointer;

    static pointer		null()
				{
				    return pointer(static_cast<T*>(nullptr));
				}
    static T*			ptr(pointer p)
				{
				    return p.get();
				}

    template <class IN_, class OUT_>
    static OUT_			copy(IN_ ib, IN_ ie, OUT_ out)
				{
				    return thrust::copy(ib, ie, out);
				}

    template <class T_>
    static void			fill(iterator ib, iterator ie, const T_& c)
				{
				    thrust::fill(ib, ie, c);
				}

    static void			init(iterator ib, iterator ie)
				{
				}
};

template <class T>
class BufTraits<T, cuda::mapped_allocator<T> >
    : public std::allocator_traits<cuda::mapped_allocator<T> >
{
  private:
    using super			= std::allocator_traits<
					cuda::mapped_allocator<T> >;

  public:
    using iterator		= typename super::pointer;
    using const_iterator	= typename super::const_pointer;

  protected:
    using pointer		= typename super::pointer;

    static T*			ptr(pointer p)
				{
				    return p.get();
				}
    
    template <class IN_, class OUT_>
    static OUT_			copy(IN_ ib, IN_ ie, OUT_ out)
				{
				    return thrust::copy(ib, ie, out);
				}

    template <class T_>
    static void			fill(iterator ib, iterator ie, const T_& c)
				{
				    thrust::fill(ib, ie, c);
				}

    static void			init(iterator ib, iterator ie)
				{
				}
};

//! 本ライブラリで定義されたクラスおよび関数を納める名前空間
namespace cuda
{
/************************************************************************
*  cuda::Array<T> and cuda::Array2<T> type aliases			*
************************************************************************/
//! 1次元CUDA配列
template <class T>
using Array = array<T, cuda::allocator<T>, 0>;
    
//! 2次元CUDA配列
template <class T>
using Array2 = array<T, cuda::allocator<T>, 0, 0>;
    
//! CUDAデバイス空間にマップされた1次元配列
template <class T>
using MappedArray = array<T, cuda::mapped_allocator<T>, 0>;
    
//! CUDAデバイス空間にマップされた2次元配列
template <class T>
using MappedArray2 = array<T, cuda::mapped_allocator<T>, 0, 0>;
    
}	// namespace cuda
}	// namespace TU
#endif	// !TU_CUDA_ARRAYPP_H
