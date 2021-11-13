/*
 *  $Id: allocator.h 1964 2016-03-29 05:56:04Z ueshiba $
 */
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

/************************************************************************
*  TU::cuda::get()							*
************************************************************************/
//! thrust::device_ptr<T> を渡されたらその raw pointer を取り出す
/*!
  cuda-9.1 まではテンプレート引数を介して thrust::device_ptr<T> を
  device関数に渡せないbugがあるので，これを回避するための関数．
 */
template <class ITER> inline const ITER&
get(const ITER& iter)
{
    return iter;
}

template <class T> inline T*
get(thrust::device_ptr<T> p)
{
    return p.get();
}

}	// namespace cuda
}	// namespace TU
