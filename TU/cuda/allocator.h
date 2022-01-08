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
  \file		allocator.h
  \brief	アロケータの定義と実装
*/
#pragma once

#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_ptr.h>

namespace TU
{
namespace cuda
{
/************************************************************************
*  class allocator<T>							*
************************************************************************/
//! CUDAデバイス上のグローバルメモリを確保するアロケータ
/*!
  \param T	確保するメモリ要素の型
*/
template <class T>
class allocator
{
  public:
    using value_type	= T;
    using pointer	= thrust::device_ptr<T>;
    using const_pointer	= thrust::device_ptr<const T>;

    template <class T_>	struct rebind	{ using other = allocator<T_>; };

  public:
    constexpr static size_t	Alignment = 256;

  public:
		allocator()					{}
    template <class T_>
		allocator(const allocator<T_>&)			{}

    pointer	allocate(std::size_t n)
		{
		  // 長さ0のメモリを要求するとCUDAのアロケータが
		  // 混乱するので，対策が必要
		    if (n == 0)
			return pointer(static_cast<T*>(nullptr));

		    auto	p = thrust::device_malloc<T>(n);
		    if (p.get() == nullptr)
			throw std::bad_alloc();
		    cudaMemset(p.get(), 0, n*sizeof(T));
		    return p;
		}
    void	deallocate(pointer p, std::size_t)
		{
		  // nullptrをfreeするとCUDAのアロケータが
		  // 混乱するので，対策が必要
		    if (p.get() != nullptr)
			thrust::device_free(p);
		}
    void	construct(T*, const value_type&)		{}
    void	destroy(T*)					{}
};

/************************************************************************
*  class mapped_ptr<T>							*
************************************************************************/
//! CUDAデバイスのメモリ領域にマップされるホスト側メモリを指すポインタ
/*!
  \param T	要素の型
*/
template <class T>
class mapped_ptr : public thrust::pointer<std::remove_cv_t<T>,
					  thrust::random_access_traversal_tag,
					  T&,
					  mapped_ptr<T> >
{
  private:
    using super		= thrust::pointer<std::remove_cv_t<T>,
					  thrust::random_access_traversal_tag,
					  T&,
					  mapped_ptr>;

  public:
    using reference	= typename super::reference;

  public:
    __host__ __device__
    mapped_ptr(T* p)			:super(p)		{}
    template <class T_> __host__ __device__
    mapped_ptr(const mapped_ptr<T_>& p)	:super(&(*p))		{}
  /*
    __host__ __device__
    reference	operator *() const
		{
		    T*	p;
		    cudaHostGetDevicePointer((void**)&p,
					     (void*)super::get(), 0);
		    return *p;
		}
  */
};

/************************************************************************
*  class mapped_allocator<T>						*
************************************************************************/
//! CUDAデバイスのメモリ領域にマップされるホスト側メモリを確保するアロケータ
/*!
  \param T	確保するメモリ要素の型
*/
template <class T>
class mapped_allocator
{
  public:
    using value_type	= T;
    using pointer	= T*;
    using const_pointer	= const T*;

  public:
		mapped_allocator()				{}
    template <class T_>
		mapped_allocator(const mapped_allocator<T_>&)	{}

    pointer	allocate(std::size_t n)
		{
		  // 長さ0のメモリを要求するとCUDAのアロケータが
		  // 混乱するので，対策が必要
		    if (n == 0)
			return pointer(static_cast<T*>(nullptr));

		    T*	p;
		    if (cudaMallocHost((void**)&p, n*sizeof(T)) != cudaSuccess)
			throw std::bad_alloc();
		    cudaMemset(p, 0, n*sizeof(T));
		    return p;
		}
    void	deallocate(pointer p, std::size_t)
		{
		  //nullptrをfreeするとCUDAのアロケータが混乱するので，対策が必要
		    if (p != nullptr)
		  	cudaFreeHost(p);
		}
};

}	// namespace cuda
}	// namespace TU
