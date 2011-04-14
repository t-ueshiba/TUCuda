/*
 *  $Id: CudaArray++.h,v 1.2 2011-04-14 08:39:34 ueshiba Exp $
 */
/*!
  \mainpage	libTUCuda++ - NVIDIA�Ђ�CUDA�𗘗p���邽�߂̃��e�B���e�B���C�u����
  \anchor	libTUCuda

  \section copyright ���쌠
  ����14-23�N�i�Ɓj�Y�ƋZ�p���������� ���쌠���L

  �n��ҁF�A�ŏr�v

  �{�v���O�����́i�Ɓj�Y�ƋZ�p�����������̐E���ł���A�ŏr�v���n�삵�C
  �i�Ɓj�Y�ƋZ�p���������������쌠�����L����閧���ł��D���쌠���L
  �҂ɂ�鋖�Ȃ��ɖ{�v���O�������g�p�C�����C���ρC��O�҂֊J������
  ���̍s�ׂ��֎~���܂��D
   
  ���̃v���O�����ɂ���Đ����邢���Ȃ鑹�Q�ɑ΂��Ă��C���쌠���L�҂�
  ��ёn��҂͐ӔC�𕉂��܂���B

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

  \section abstract �T�v
  libTUCuda++�́CC++���ɂ�����NVIDIA�Ђ�CUDA�𗘗p���邽�߂̃��e�B���e�B
  ���C�u�����ł���D�ȉ��̂悤�ȃN���X����ъ֐�����������Ă���D

  <b>�f�o�C�X���̃O���[�o���������̈�ɂƂ���1���������2�����z��</b>
  - #TU::CudaArray
  - #TU::CudaArray2
  
  <b>�t�B���^�����O
  - #TU::CudaFilter2
  - #TU::CudaGaussianConvolver2

  <b>�f�o�C�X���̃R���X�^���g�������̈�ւ̃R�s�[</b>
  - #TU::cudaCopyToConstantMemory

  \file		CudaArray++.h
  \brief	��{�I�ȃf�[�^�^���O���[�o���Ȗ��O��Ԃɒǉ�
*/
#ifndef __TUCudaArrayPP_h
#define __TUCudaArrayPP_h

#include <thrust/device_new.h>
#include <thrust/device_delete.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include "TU/Array++.h"

/*!
  \namespace	TU
  \brief	�{���C�u�����Œ�`���ꂽ�N���X����ъ֐���[�߂閼�O���
*/
namespace TU
{
/************************************************************************
*  class CudaBuf<T>							*
************************************************************************/
//! CUDA�ɂ����ăf�o�C�X���Ɋm�ۂ����ϒ��o�b�t�@�N���X
/*!
  �P�ƂŎg�p���邱�Ƃ͂Ȃ��C#TU::Array�܂���#TU::Array2��
  ��2�e���v���[�g�����Ɏw�肷�邱�Ƃɂ���āC�����̊��N���X�Ƃ��Ďg���D
  \param T	�v�f�̌^
*/
template <class T>
class CudaBuf
{
  public:
  //! �v�f�̌^    
    typedef T						value_type;
  //! �v�f�ւ̎Q��    
    typedef thrust::device_reference<value_type>	reference;
  //! �v�f�ւ̎Q��    
    typedef thrust::device_reference<const value_type>	const_reference;
  //! �v�f�ւ̃|�C���^    
    typedef thrust::device_ptr<value_type>		pointer;
  //! �v�f�ւ̃|�C���^    
    typedef thrust::device_ptr<const value_type>	const_pointer;
    
  public:
    explicit CudaBuf(u_int siz=0)				;
    CudaBuf(pointer p, u_int siz)				;
    CudaBuf(const CudaBuf& b)					;
    CudaBuf&		operator =(const CudaBuf& b)		;
    ~CudaBuf()							;

    pointer		ptr()					;
    const_pointer	ptr()				const	;
    size_t		size()				const	;
    bool		resize(u_int siz)			;
    void		resize(pointer p, u_int siz)		;
    static u_int	stride(u_int siz)			;
    std::istream&	get(std::istream& in, u_int m=0)	;
    std::ostream&	put(std::ostream& out)		const	;
    
  private:
    static pointer	memalloc(u_int siz)			;
    static void		memfree(pointer p, u_int siz)		;

  private:
    u_int	_size;		// the number of elements in the buffer
    pointer	_p;		// pointer to the buffer area
    bool	_shared;	// buffer area is shared with other object
};
    
//! �w�肵���v�f���̃o�b�t�@�𐶐�����D
/*!
  \param siz	�v�f��
*/
template <class T> inline
CudaBuf<T>::CudaBuf(u_int siz)
    :_size(siz), _p(memalloc(_size)), _shared(false)
{
}

//! �O���̗̈�Ɨv�f�����w�肵�ăo�b�t�@�𐶐�����D
/*!
  \param p	�O���̈�ւ̃|�C���^
  \param siz	�v�f��
*/
template <class T> inline
CudaBuf<T>::CudaBuf(pointer p, u_int siz)
    :_size(siz), _p(p), _shared(true)
{
}
    
//! �R�s�[�R���X�g���N�^
template <class T> inline
CudaBuf<T>::CudaBuf(const CudaBuf<T>& b)
    :_size(b._size), _p(memalloc(_size)), _shared(false)
{
    thrust::copy(b.ptr(), b.ptr() + b.size(), ptr());
}

//! �W��������Z�q
template <class T> inline CudaBuf<T>&
CudaBuf<T>::operator =(const CudaBuf<T>& b)
{
    if (this != &b)
    {
	resize(b._size);
	thrust::copy(b.ptr(), b.ptr() + b.size(), ptr());
    }
    return *this;
}

//! �f�X�g���N�^
template <class T> inline
CudaBuf<T>::~CudaBuf()
{
    if (!_shared)
	memfree(_p, _size);
}
    
//! �o�b�t�@���g�p��������L���̈�ւ̃|�C���^��Ԃ��D
template <class T> inline typename CudaBuf<T>::pointer
CudaBuf<T>::ptr()
{
    return _p;
}

//! �o�b�t�@���g�p��������L���̈�ւ̃|�C���^��Ԃ��D
template <class T> inline typename CudaBuf<T>::const_pointer
CudaBuf<T>::ptr() const
{
    return _p;
}
    
//! �o�b�t�@�̗v�f����Ԃ��D
template <class T> inline size_t
CudaBuf<T>::size() const
{
    return _size;
}
    
//! �o�b�t�@�̗v�f����ύX����D
/*!
  �������C���̃I�u�W�F�N�g�ƋL���̈�����L���Ă���o�b�t�@�̗v�f����
  �ύX���邱�Ƃ͂ł��Ȃ��D
  \param siz			�V�����v�f��
  \return			siz�����̗v�f�������傫�����true�C����
				�łȂ����false
  \throw std::logic_error	�L���̈�𑼂̃I�u�W�F�N�g�Ƌ��L���Ă���ꍇ
				�ɑ��o
*/
template <class T> bool
CudaBuf<T>::resize(u_int siz)
{
    if (_size == siz)
	return false;
    
    if (_shared)
	throw std::logic_error("CudaBuf<T>::resize: cannot change size of shared buffer!");

    memfree(_p, _size);
    const u_int	old_size = _size;
    _size = siz;
    _p = memalloc(_size);

    return _size > old_size;
}

//! �o�b�t�@�������Ŏg�p����L���̈���w�肵�����̂ɕύX����D
/*!
  \param p	�V�����L���̈�ւ̃|�C���^
  \param siz	�V�����v�f��
*/
template <class T> inline void
CudaBuf<T>::resize(pointer p, u_int siz)
{
    if (!_shared)
	memfree(_p, _size);
    _size = siz;
    _p = p;
    _shared = true;
}

//! �w�肳�ꂽ�v�f�������L���̈���m�ۂ��邽�߂Ɏ��ۂɕK�v�ȗv�f����Ԃ��D
/*!
  �i�L���e�ʂł͂Ȃ��j�v�f����16�܂���32�̔{���ɂȂ�悤�C
  �^����ꂽ�v�f�����J��グ��D
  \param siz	�v�f��
  \return	16�܂���32�̔{���ɌJ��グ��ꂽ�v�f��
*/
template <class T> inline u_int
CudaBuf<T>::stride(u_int siz)
{
    const u_int	ALIGN = ((sizeof(T) == 8) || (sizeof(T) == 16) ? 16 : 32);
	
    return (siz > 0 ? ALIGN * ((siz - 1) / ALIGN + 1) : 0);
}
    
//! ���̓X�g���[������w�肵���ӏ��ɔz���ǂݍ���(ASCII)�D
/*!
  \param in	���̓X�g���[��
  \param m	�ǂݍ��ݐ�̐擪���w�肷��index
  \return	in�Ŏw�肵�����̓X�g���[��
*/
template <class T> std::istream&
CudaBuf<T>::get(std::istream& in, u_int m)
{
    const u_int	BufSiz = (sizeof(T) < 2048 ? 2048 / sizeof(T) : 1);
    T* const	tmp = new T[BufSiz];
    u_int	n = 0;
    
    while (n < BufSiz)
    {
	char	c;
	while (in.get(c))		// skip white spaces other than '\n'
	    if (!isspace(c) || c == '\n')
		break;

	if (!in || c == '\n')
	{
	    resize(m + n);
	    break;
	}

	in.putback(c);
	in >> tmp[n++];
    }
    if (n == BufSiz)
	get(in, m + n);

    for (u_int i = 0; i < n; ++i)
	_p[m + i] = tmp[i];

    delete [] tmp;
    
    return in;
}

//! �o�̓X�g���[���ɔz��������o��(ASCII)�D
/*!
  \param out	�o�̓X�g���[��
  \return	out�Ŏw�肵���o�̓X�g���[��
*/
template <class T> std::ostream&
CudaBuf<T>::put(std::ostream& out) const
{
    for (u_int i = 0; i < _size; )
	out << ' ' << _p[i++];
    return out;
}

template <class T> inline typename CudaBuf<T>::pointer
CudaBuf<T>::memalloc(u_int siz)
{
    if (siz > 0)
    {
	pointer	p = thrust::device_new<T>(siz);
	cudaMemset(p.get(), 0, sizeof(T) * siz);
	return p;
    }
    else
	return pointer((T*)0);
}
    
template <class T> inline void
CudaBuf<T>::memfree(pointer p, u_int siz)
{
    if (p.get() != 0)
	thrust::device_delete(p, siz);
}
    
/************************************************************************
*  class CudaArray<T>							*
************************************************************************/
//! CUDA�ɂ����ăf�o�C�X���Ɋm�ۂ����T�^�I�u�W�F�N�g��1�����z��N���X
/*!
  \param T	�v�f�̌^
*/
template <class T>
class CudaArray : public Array<T, CudaBuf<T> >
{
  private:
    typedef Array<T, CudaBuf<T> >		super;

  public:
  //! �o�b�t�@�̌^
    typedef typename super::buf_type		buf_type;
  //! �v�f�̌^    
    typedef typename super::value_type		value_type;
  //! �v�f�ւ̎Q��
    typedef typename super::reference		reference;
  //! �萔�v�f�ւ̎Q��
    typedef typename super::const_reference	const_reference;
  //! �v�f�ւ̃|�C���^
    typedef typename super::pointer		pointer;
  //! �萔�v�f�ւ̃|�C���^
    typedef typename super::const_pointer	const_pointer;
  //! �����q
    typedef typename super::iterator		iterator;
  //! �萔�����q
    typedef typename super::const_iterator	const_iterator;
  //! �t�����q    
    typedef typename super::reverse_iterator	reverse_iterator;
  //! �萔�t�����q    
    typedef typename super::const_reverse_iterator
						const_reverse_iterator;
  //! �|�C���^�Ԃ̍�
    typedef typename super::difference_type	difference_type;
  //! �v�f�ւ̒��ڃ|�C���^
    typedef value_type*				raw_pointer;
  //! �萔�v�f�ւ̒��ڃ|�C���^
    typedef const value_type*			const_raw_pointer;
    
  public:
    CudaArray()							;
    explicit CudaArray(u_int d)					;
    template <class B>
    CudaArray(const Array<T, B>& a)				;
    template <class B>
    CudaArray&	operator =(const Array<T, B>& a)		;
    template <class B> const CudaArray&
		write(Array<T, B>& a)			const	;
    CudaArray&	operator =(const value_type& c)			;

		operator raw_pointer()				;
		operator const_raw_pointer()		const	;
    
    using	super::begin;
    using	super::end;
    using	super::rbegin;
    using	super::rend;
    using	super::size;
    using	super::dim;
    using	super::resize;
};

//! CUDA�z��𐶐�����D
template <class T> inline
CudaArray<T>::CudaArray()
    :super()
{
}

//! �w�肵���v�f����CUDA�z��𐶐�����D
/*!
  \param d	�z��̗v�f��
*/
template <class T> inline
CudaArray<T>::CudaArray(u_int d)
    :super(d)
{
}

//! ���̔z��Ɠ���v�f������CUDA�z������i�R�s�[�R���X�g���N�^�̊g���j
/*!
  �R�s�[�R���X�g���N�^�͕ʓr�����I�ɐ��������D
  \param a	�R�s�[���̔z��
*/
template <class T> template <class B> inline
CudaArray<T>::CudaArray(const Array<T, B>& a)
    :super(a.size())
{
    thrust::copy(a.begin(), a.end(), begin());
}

//! ���̔z��������ɑ������i�W��������Z�q�̊g���j
/*!
  �W��������Z�q�͕ʓr�����I�ɐ��������D
  \param a	�R�s�[���̔z��
  \return	���̔z��
*/
template <class T> template <class B> inline CudaArray<T>&
CudaArray<T>::operator =(const Array<T, B>& a)
{
    resize(a.size());
    thrust::copy(a.begin(), a.end(), begin());
    return *this;
}

//! ����CUDA�z��̓��e�𑼂̔z��ɏ����o���D
/*!
  \param a	�R�s�[��̔z��
  \return	���̔z��
*/
template <class T> template <class B> inline const CudaArray<T>&
CudaArray<T>::write(Array<T, B>& a) const
{
    a.resize(size());
    thrust::copy(begin(), end(), a.begin());
    return *this;
}

//! �S�Ă̗v�f�ɓ���̒l��������D
/*!
  \param c	�������l
  \return	���̔z��
*/
template <class T> inline CudaArray<T>&
CudaArray<T>::operator =(const value_type& c)
{
    thrust::fill(begin(), end(), c);
    return *this;
}

//! �z��̓����L���̈�ւ̃|�C���^��Ԃ��D
/*!
  \return	�����L���̈�ւ̃|�C���^
*/
template <class T> inline
CudaArray<T>::operator raw_pointer()
{
    return super::operator pointer().get();
}
		    
//! �z��̓����L���̈�ւ̃|�C���^��Ԃ��D
/*!
  \return	�����L���̈�ւ̃|�C���^
*/
template <class T> inline
CudaArray<T>::operator const_raw_pointer() const
{
    return super::operator const_pointer().get();
}
		    
/************************************************************************
*  class CudaArray2<T>							*
************************************************************************/
//! CUDA�ɂ����ăf�o�C�X���Ɋm�ۂ����T�^�I�u�W�F�N�g��2�����z��N���X
/*!
  \param T	�v�f�̌^
*/
template <class T>
class CudaArray2 : public Array2<CudaArray<T>, CudaBuf<T> >
{
  private:
    typedef Array2<CudaArray<T>, CudaBuf<T> >	super;
    
  public:
  //! �s�o�b�t�@�̌^
    typedef typename super::row_buf_type	row_buf_type;
  //! �s�̌^    
    typedef typename super::row_type		row_type;
  //! �s�ւ̎Q��    
    typedef typename super::row_reference	row_reference;
  //! �萔�s�ւ̎Q��    
    typedef typename super::row_const_reference	row_const_reference;
  //! �s�ւ̃|�C���^    
    typedef typename super::row_pointer		row_pointer;
  //! �萔�s�ւ̃|�C���^    
    typedef typename super::row_const_pointer	row_const_pointer;
  //! �s�̔����q    
    typedef typename super::row_iterator	row_iterator;
  //! �s�̒萔�����q    
    typedef typename super::row_const_iterator	row_const_iterator;
  //! �s�̋t�����q    
    typedef typename super::row_reverse_iterator
						row_reverse_iterator;
  //! �s�̒萔�t�����q    
    typedef typename super::row_const_reverse_iterator
						row_const_reverse_iterator;
  //! �o�b�t�@�̌^    
    typedef typename super::buf_type		buf_type;
  //! �v�f�̌^    
    typedef typename super::value_type		value_type;
  //! �v�f�ւ̎Q��    
    typedef typename super::reference		reference;
  //! �萔�v�f�ւ̎Q��    
    typedef typename super::const_reference	const_reference;
  //! �v�f�ւ̃|�C���^    
    typedef typename super::pointer		pointer;
  //! �萔�v�f�ւ̃|�C���^    
    typedef typename super::const_pointer	const_pointer;
  //! �|�C���^�Ԃ̍�    
    typedef typename super::difference_type	difference_type;
  //! �v�f�ւ̒��ڃ|�C���^
    typedef value_type*				raw_pointer;
  //! �萔�v�f�ւ̒��ڃ|�C���^
    typedef const value_type*			const_raw_pointer;

  public:
    CudaArray2()						;
    CudaArray2(u_int r, u_int c)				;
    template <class T2, class B2, class R2>
    CudaArray2(const Array2<T2, B2, R2>& a)			;
    template <class T2, class B2, class R2>
    CudaArray2&	operator =(const Array2<T2, B2, R2>& a)		;
    template <class T2, class B2, class R2> const CudaArray2&
		write(Array2<T2, B2, R2>& a)		const	;
    CudaArray2&	operator =(const value_type& c)			;
		operator raw_pointer()				;
		operator const_raw_pointer()		const	;

    using	super::begin;
    using	super::end;
    using	super::size;
    using	super::dim;
    using	super::nrow;
    using	super::ncol;
    using	super::stride;
};

//! 2����CUDA�z��𐶐�����D
template <class T> inline
CudaArray2<T>::CudaArray2()
    :super()
{
}

//! �s���Ɨ񐔂��w�肵��2����CUDA�z��𐶐�����D
/*!
  \param r	�s��
  \param c	��
*/
template <class T> inline
CudaArray2<T>::CudaArray2(u_int r, u_int c)
    :super(r, c)
{
}

//! ���̔z��Ɠ���v�f������CUDA�z������i�R�s�[�R���X�g���N�^�̊g���j
/*!
  �R�s�[�R���X�g���N�^�͕ʓr�����I�ɐ��������D
  \param a	�R�s�[���̔z��
*/
template <class T>
template <class T2, class B2, class R2> inline
CudaArray2<T>::CudaArray2(const Array2<T2, B2, R2>& a)
    :super()
{
    operator =(a);
}

//! ���̔z��������ɑ������i�W��������Z�q�̊g���j
/*!
  �W��������Z�q�͕ʓr�����I�ɐ��������D
  \param a	�R�s�[���̔z��
  \return	���̔z��
*/
template <class T>
template <class T2, class B2, class R2> inline CudaArray2<T>&
CudaArray2<T>::operator =(const Array2<T2, B2, R2>& a)
{
    resize(a.nrow(), a.ncol());
    if (a.nrow() > 0 && a.stride() == stride())
    {
	thrust::copy(a[0].begin(), a[a.nrow()-1].end(), (*this)[0].begin());
    }
    else
    {
	for (u_int i = 0; i < nrow(); ++i)
	    (*this)[i] = a[i];
    }
    return *this;
}

//! ����CUDA�z��̓��e�𑼂̔z��ɏ����o���D
/*!
  \param a	�R�s�[��̔z��
  \return	���̔z��
*/
template <class T>
template <class T2, class B2, class R2> inline const CudaArray2<T>&
CudaArray2<T>::write(Array2<T2, B2, R2>& a) const
{
    a.resize(nrow(), ncol());
    if (nrow() > 0 && stride() == a.stride())
    {
	thrust::copy((*this)[0].begin(),
		     (*this)[nrow()-1].end(), a[0].begin());
    }
    else
    {
	for (u_int i = 0; i < nrow(); ++i)
	    (*this)[i].write(a[i]);
    }
    return *this;
}

//! �S�Ă̗v�f�ɓ���̒l��������D
/*!
  \param c	�������l
  \return	���̔z��
*/
template <class T> inline CudaArray2<T>&
CudaArray2<T>::operator =(const value_type& c)
{
    if (nrow() > 0)
	thrust::fill((*this)[0].begin(), (*this)[nrow()-1].end(), c);
    return *this;
}

//! �z��̓����L���̈�ւ̃|�C���^��Ԃ��D
/*!
  
  \return	�����L���̈�ւ̃|�C���^
*/
template <class T> inline
CudaArray2<T>::operator raw_pointer()
{
    return super::operator pointer().get();
}
		    
//! �z��̓����L���̈�ւ̃|�C���^��Ԃ��D
/*!
  \return	�����L���̈�ւ̃|�C���^
*/
template <class T> inline
CudaArray2<T>::operator const_raw_pointer() const
{
    return super::operator const_pointer().get();
}

/************************************************************************
*  utilities								*
************************************************************************/
//! CUDA�̒萔�������̈�Ƀf�[�^���R�s�[����D
/*!
  \param begin	�R�s�[���f�[�^�̐擪���w�������q
  \param end	�R�s�[���f�[�^�̖����̎����w�������q
  \param dst	�R�s�[��̒萔�������̈���w���|�C���^
*/
template <class Iterator, class T> inline void
cudaCopyToConstantMemory(Iterator begin, Iterator end, T* dst)
{
    typedef typename std::iterator_traits<Iterator>	iterator_traits;

    if (begin < end)
	cudaMemcpyToSymbol((const char*)dst, &(*begin),
			   (end - begin)*sizeof(T));
}

}
#endif	/* !__TUCudaArrayPP_h */
