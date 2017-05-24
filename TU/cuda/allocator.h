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
#include <boost/operators.hpp>

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
    using pointer	= thrust::device_ptr<value_type>;
    using const_pointer	= thrust::device_ptr<const value_type>;
  //using reference		= thrust::device_reference<value_type>;
  //using const_reference	= thrust::device_reference<const value_type>;

    template <class T_>	struct rebind	{ typedef allocator<T_> other; };

  public:
		allocator()					{}
    template <class U>
		allocator(const allocator<U>&)			{}

    pointer	allocate(std::size_t n)
		{
		  // 長さ0のメモリを要求するとCUDAのアロケータが
		  // 混乱するので，対策が必要
		    if (n == 0)
			return pointer((value_type*)nullptr);

		    auto	p = thrust::device_malloc<value_type>(n);
		    if (p.get() == nullptr)
			throw std::bad_alloc();
		    cudaMemset(p.get(), 0, n*sizeof(value_type));
		    return p;
		}
    void	deallocate(pointer p, std::size_t)
		{
		  // nullptrをfreeするとCUDAのアロケータが
		  // 混乱するので，対策が必要
		    if (p.get() != nullptr)
			thrust::device_free(p);
		}
    void	construct(pointer, const value_type&)		{}
    void	destroy(pointer)				{}
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
  //using reference	= typename std::iterator_traits<pointer>::reference;
  //using const_reference	= typename std::iterator_traits<const_pointer>
  //					      ::reference;

  public:
		mapped_allocator()				{}
    template <class U>
		mapped_allocator(const mapped_allocator<U>&)	{}

    pointer	allocate(std::size_t n)
		{
		  // 長さ0のメモリを要求するとCUDAのアロケータが
		  // 混乱するので，対策が必要
		    if (n == 0)
			return pointer((value_type*)nullptr);

		    value_type*	q;
		    if (cudaHostAlloc((void**)&q, n*sizeof(value_type),
				      cudaHostAllocMapped)
			!= cudaSuccess)
			throw std::bad_alloc();
		    pointer	p(q);
		    cudaMemset(p.get(), 0, n*sizeof(value_type));
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
