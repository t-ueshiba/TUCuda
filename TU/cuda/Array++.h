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

  <b>デバイス側のグローバルメモリ領域にとられる1次元および2次元配列</b>
  - #TU::CudaArray
  - #TU::CudaArray2

  <b>デバイス側のテクスチャメモリ</b>
  - #TU::cuda::Texture
  
  <b>フィルタリング</b>
  - #TU::cuda::FIRFilter2
  - #TU::cuda::FIRGaussianConvolver2

  <b>ユティリティ</b>
  - #TU::cuda::copyToConstantMemory()
  - #TU::cuda::subsample()
  - #TU::cuda::op3x3()
  - #TU::cuda::suppressNonExtrema3x3()
  
  \file		Array++.h
  \brief	CUDAデバイス上の配列に関連するクラスの定義と実装
*/
#ifndef __TU_CUDA_ARRAYPP_H
#define __TU_CUDA_ARRAYPP_H

#include <thrust/device_allocator.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include "TU/Array++.h"

//! 植芝によって開発されたクラスおよび関数を納める名前空間
namespace TU
{
//! 本ライブラリで定義されたクラスおよび関数を納める名前空間
namespace cuda
{
/************************************************************************
*  class allocator<T>							*
************************************************************************/
template <class T>
class allocator
{
  public:
    typedef T						value_type;
    typedef thrust::device_ptr<value_type>		pointer;
    typedef thrust::device_ptr<const value_type>	const_pointer;
    typedef thrust::device_reference<value_type>	reference;
    typedef thrust::device_reference<const value_type>	const_reference;
    typedef size_t					size_type;
    typedef ptrdiff_t					difference_type;

    template <class T_>	struct rebind	{ typedef allocator<T_> other; };

  public:
			allocator()					{}
    template <class U>	allocator(const allocator<U>&)			{}

    static pointer	allocate(size_type n,
				 typename std::allocator<void>
					     ::const_pointer=nullptr)
			{
			  // 長さ0のメモリを要求するとCUDAのアロケータの動作が
			  // 混乱するので，対策が必要
			    if (n == 0)
				return pointer((value_type*)nullptr);

			    pointer	p = thrust::
					    device_malloc<value_type>(n);
			    if (p.get() == nullptr)
				throw std::bad_alloc();
			    cudaMemset(p.get(), 0, n*sizeof(value_type));
			    return p;
			}
    static void		deallocate(pointer p, size_type)
			{
			  // nullポインタをfreeするとCUDAのアロケータの動作が
			  // 混乱するので，対策が必要
			    if (p.get() != nullptr)
				thrust::device_free(p);
			}
    static void		construct(pointer, const value_type&)		{}
    static void		destroy(pointer)				{}
    constexpr
    static size_type	max_size()
			{
			    return std::numeric_limits<size_type>::max()
				 / sizeof(value_type);
			}
};
}	// namespace cuda
    
/************************************************************************
*  specialization for Buf<T, D, ALLOC> for CUDA				*
************************************************************************/
template <class T>
struct BufTraits<T, cuda::allocator<T> >
{
    typedef cuda::allocator<T>				allocator_type;
    typedef typename allocator_type::pointer		iterator;
    typedef typename allocator_type::const_pointer	const_iterator;

  protected:
    template <class IN_, class OUT_>
    static OUT_	copy(IN_ ib, IN_ ie, OUT_ out)
		{
		    return thrust::copy(ib, ie, out);
		}
    template <class T_>
    static void	fill(iterator ib, iterator ie, const T_& c)
		{
		    thrust::fill(ib, ie, c);
		}
    static void	init(iterator ib, iterator ie)
		{
		}
};

/************************************************************************
*  CudaArray<T> and CudaArray2<T> type aliases				*
************************************************************************/
//! 1次元CUDA配列
template <class T>
using CudaArray = Array<T, 0, cuda::allocator<T> >;
    
//! 2次元CUDA配列
template <class T>
using CudaArray2 = Array2<CudaArray<T> >;

}	// namespace TU
#endif	// !__TU_CUDA_ARRAYPP_H
