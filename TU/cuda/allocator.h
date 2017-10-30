/*
 *  $Id: allocator.h 1964 2016-03-29 05:56:04Z ueshiba $
 */
/*!
  \file		allocator.h
  \brief	アロケータの定義と実装
*/
#ifndef TU_CUDA_ALLOCATOR_H
#define TU_CUDA_ALLOCATOR_H

#include <thrust/device_allocator.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <boost/operators.hpp>

namespace TU
{
/************************************************************************
*  algorithms overloaded for thrust::pointer, thrust::const_pointer	*
************************************************************************/
template <size_t N, class S, class T> inline void
copy(thrust::device_ptr<S> p, size_t n, thrust::device_ptr<T> q)
{
    thrust::copy_n(p, (N ? N : n), q);
}
    
template <size_t N, class S, class T> inline void
copy(thrust::device_ptr<S> p, size_t n, T* q)
{
    thrust::copy_n(p, (N ? N : n), q);
}
    
template <size_t N, class S, class T> inline void
copy(const S* p, size_t n, thrust::device_ptr<T> q)
{
    thrust::copy_n(p, (N ? N : n), q);
}

template <size_t N, class T, class S> inline void
fill(thrust::device_ptr<T> q, size_t n, const S& val)
{
    thrust::fill_n(q, (N ? N : n), val);
}

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
class mapped_ptr : public boost::random_access_iterator_helper<mapped_ptr<T>, T>
{
  private:
    using super		= boost::random_access_iterator_helper<mapped_ptr, T>;
	
  public:
    using value_type	= typename std::remove_cv<T>::type;
    using		typename super::difference_type;
	
  public:
    mapped_ptr(T* p)			:_p(p)		{}
    template <class T_>
    mapped_ptr(const mapped_ptr<T_>& p)	:_p(&(*p))	{}

    T&		operator *()			const	{ return *_p; }
    T*		get() const
		{
		    T*	p;
		    cudaHostGetDevicePointer((void**)&p, (void*)_p, 0);
		    return p;
		}
    __host__ __device__
    mapped_ptr&	operator ++()				{ ++_p; return *this; }
    __host__ __device__
    mapped_ptr&	operator --()				{ --_p; return *this; }
    __host__ __device__
    mapped_ptr&	operator +=(difference_type d)		{_p += d; return *this;}
    __host__ __device__
    mapped_ptr&	operator -=(difference_type d)		{_p -= d; return *this;}
    __host__ __device__ difference_type
		operator - (const mapped_ptr& p) const	{ return _p - p._p; }
    __host__ __device__
    bool	operator ==(const mapped_ptr& p) const	{ return _p == p._p; }
    __host__ __device__
    bool	operator < (const mapped_ptr& p) const	{ return _p < p._p; }

  private:
    T*		_p;
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
    using pointer	= mapped_ptr<T>;
    using const_pointer	= mapped_ptr<const T>;

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

		    T*	q;
		    if (cudaHostAlloc((void**)&q, n*sizeof(T),
				      cudaHostAllocMapped) != cudaSuccess)
			throw std::bad_alloc();
		    pointer	p(q);
		    cudaMemset(p.get(), 0, n*sizeof(T));
		    return p;
		}
    void	deallocate(pointer p, std::size_t)
		{
		  // nullptrをfreeするとCUDAのアロケータが混乱するので，対策が必要
		    if (p.get() != nullptr)
			cudaFreeHost(p.get());
		}
};

}	// namespace cuda
}	// namespace TU
#endif	// !TU_CUDA_ALLOCATOR_H
