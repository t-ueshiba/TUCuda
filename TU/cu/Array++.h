/*!
  \mainpage	libTUCuda - NVIDIA社のCUDAを利用するためのユティリティライブラリ
  \anchor	libTUCuda

  \section copyright 著作権
  Software License Agreement (BSD License)

  Copyright (c) 2021, National Institute of Advanced Industrial Science and Technology (AIST)
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     disclaimer in the documentation and/or other materials provided
     with the distribution.
   * Neither the name of National Institute of Advanced Industrial
     Science and Technology (AIST) nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
  POSSIBILITY OF SUCH DAMAGE.

  Author: Toshio Ueshiba


  \section abstract 概要
  libTUCudaは，C++環境においてNVIDIA社のCUDAを利用するためのユティリティ
  ライブラリである．以下のようなクラスおよび関数が実装されている．

  <b>デバイス側のグローバルメモリを確保するアロケータ
  - #TU::cu::allocator

  <b>デバイス側のグローバルメモリ領域にマップされるホスト側メモリを確保するアロケータ
  - #TU::cu::mapped_allocator

  <b>デバイス側のグローバルメモリ領域にとられる1/2/3次元配列</b>
  - #TU::cu::Array
  - #TU::cu::Array2
  - #TU::cu::Array3

  <b>デバイス側のグローバルメモリ領域にマップされるホスト側1/2/3次元配列</b>
  - #TU::cu::MappedArray
  - #TU::cu::MappedArray2
  - #TU::cu::MappedArray3

  <b>デバイス側のテクスチャメモリ</b>
  - #TU::cu::Texture

  <b>フィルタリング</b>
  - #TU::cu::FIRFilter2
  - #TU::cu::FIRGaussianConvolver2
  - #TU::cu::BoxFilter2
  - #TU::cu::GuidedFilter2

  <b>アルゴリズム</b>
  - #TU::cu::copyToConstantMemory(ITER, ITER, T*)
  - #TU::cu::subsample(IN, IN, OUT)
  - #TU::cu::op3x3(IN, IN, OUT, OP)

  <b>時間計測</b>
  - #TU::cu::clock

  \file		Array++.h
  \brief	CUDAデバイス上の配列に関連するクラスの定義と実装
*/
#pragma once

#include <thrust/copy.h>
#include <thrust/fill.h>
#include "TU/cu/allocator.h"
#include "TU/cu/iterator.h"
#include "TU/Array++.h"

//! thrust::device_ptr<S> 型の引数に対しADLによって名前解決を行う関数を収める名前空間
namespace thrust
{
/************************************************************************
*  algorithms overloaded for thrust::device_ptr<T>			*
************************************************************************/
template <size_t N, class S, class T> inline void
copy(device_ptr<S> p, size_t n, device_ptr<T> q)
{
    copy_n(p, (N ? N : n), q);
}

template <size_t N, class S, class T> void
copy(const S* p, size_t n, device_ptr<T> q)
{
    copy_n(p, (N ? N : n), q);
}

template <size_t N, class S, class T> inline void
copy(device_ptr<S> p, size_t n, T* q)
{
    copy_n(p, (N ? N : n), q);
}

template <size_t N, class T, class S> inline void
fill(device_ptr<T> q, size_t n, const S& val)
{
    fill_n(q, (N ? N : n), val);
}

}	// namespace thrust

//! 植芝によって開発されたクラスおよび関数を納める名前空間
namespace TU
{
/************************************************************************
*  specialization for BufTraits<T, ALLOC> for CUDA			*
************************************************************************/
template <class T>
class BufTraits<T, cu::allocator<T> >
    : public std::allocator_traits<cu::allocator<T> >
{
  private:
    using super			= std::allocator_traits<cu::allocator<T> >;

  public:
    using iterator		= typename super::pointer;
    using const_iterator	= typename super::const_pointer;

  protected:
    using pointer		= typename super::pointer;

    constexpr static size_t	Alignment = super::allocator_type::Alignment;

    static pointer		null()
				{
				    return pointer(static_cast<T*>(nullptr));
				}
    static T*			ptr(pointer p)
				{
				    return p.get();
				}
};

template <class T>
class BufTraits<T, cu::mapped_allocator<T> >
    : public std::allocator_traits<cu::mapped_allocator<T> >
{
  private:
    using super			= std::allocator_traits<
					cu::mapped_allocator<T> >;

  public:
    using iterator		= typename super::pointer;
    using const_iterator	= typename super::const_pointer;

  protected:
    using pointer		= typename super::pointer;

    constexpr static size_t	Alignment = 0;

    static pointer		null()
				{
				    return static_cast<T*>(nullptr);
				}
    static T*			ptr(pointer p)
				{
				    return p;
				}
};

//! 本ライブラリで定義されたクラスおよび関数を納める名前空間
namespace cu
{
/************************************************************************
*  cu::Array<T> and cu::Array2<T> type aliases			*
************************************************************************/
//! 1次元CUDA配列
template <class T>
using Array = array<T, cu::allocator<T>, 0>;

//! 2次元CUDA配列
template <class T>
using Array2 = array<T, cu::allocator<T>, 0, 0>;

//! 3次元CUDA配列
template <class T>
using Array3 = array<T, cu::allocator<T>, 0, 0, 0>;

//! CUDAデバイス空間にマップされた1次元配列
template <class T>
using MappedArray = array<T, cu::mapped_allocator<T>, 0>;

//! CUDAデバイス空間にマップされた2次元配列
template <class T>
using MappedArray2 = array<T, cu::mapped_allocator<T>, 0, 0>;

//! CUDAデバイス空間にマップされた3次元配列
template <class T>
using MappedArray3 = array<T, cu::mapped_allocator<T>, 0, 0, 0>;

}	// namespace cu
}	// namespace TU
