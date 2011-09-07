/*
 *  ����21-22�N�i�Ɓj�Y�ƋZ�p���������� ���쌠���L
 *  
 *  �n��ҁF�A�ŏr�v
 *
 *  �{�v���O�����́i�Ɓj�Y�ƋZ�p�����������̐E���ł���A�ŏr�v���n�삵�C
 *  �i�Ɓj�Y�ƋZ�p���������������쌠�����L����閧���ł��D���쌠���L
 *  �҂ɂ�鋖�Ȃ��ɖ{�v���O�������g�p�C�����C���ρC��O�҂֊J������
 *  ���̍s�ׂ��֎~���܂��D
 *  
 *  ���̃v���O�����ɂ���Đ����邢���Ȃ鑹�Q�ɑ΂��Ă��C���쌠���L�҂�
 *  ��ёn��҂͐ӔC�𕉂��܂���B
 *
 *  Copyright 2009-2010.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *
 *  $Id: SparseMatrix++.h,v 1.2 2011-09-07 05:00:28 ueshiba Exp $
 */
/*!
  \file		SparseMatrix++.h
  \brief	�N���X TU::SparseMatrix �̒�`�Ǝ���
*/
#ifndef __TUSparseMatrixPP_h
#define __TUSparseMatrixPP_h

#include "TU/Vector++.h"
#include <vector>
#include <algorithm>
#include <functional>
#include <mkl_pardiso.h>

namespace TU
{
/************************************************************************
*  class SparseMatrix<T, SYM>						*
************************************************************************/
//! Intel Math-Kernel Library(MKL)�̃t�H�[�}�b�g�ɂ��a�s��
template <class T, bool SYM=false>
class SparseMatrix
{
  public:
    typedef T		value_type;		//!< �����̌^

  public:
  // �\������
    void		beginInit()					;
    void		setRow()					;
    void		setCol(u_int col)				;
    void		setCol(u_int col, value_type val)		;
    void		copyRow()					;
    void		endInit()					;
    
  // ��{���
    u_int		dim()					const	;
    u_int		nrow()					const	;
    u_int		ncol()					const	;
    u_int		nelements()				const	;
    u_int		nelements(u_int i)			const	;
    T			operator ()(u_int i, u_int j)		const	;
    T&			operator ()(u_int i, u_int j)			;

  // ��{���Z
    SparseMatrix&	operator  =(value_type c)			;
    SparseMatrix&	operator *=(value_type c)			;
    SparseMatrix&	operator /=(value_type c)			;
    SparseMatrix&	operator +=(const SparseMatrix& A)		;
    SparseMatrix&	operator -=(const SparseMatrix& A)		;
    SparseMatrix	operator  +(const SparseMatrix& A)	const	;
    SparseMatrix	operator  -(const SparseMatrix& A)	const	;
    template <class S, class B>
    Vector<T>		operator  *(const Vector<S, B>& v)	const	;
    SparseMatrix<T, true>
			compose()				const	;
    SparseMatrix<T, true>
			compose(const SparseMatrix<T, true>& W)	const	;

  // �u���b�N���Z
    Vector<T>		operator ()(u_int i, u_int j, u_int d)	const	;
    Matrix<T>		operator ()(u_int i, u_int j,
				    u_int r, u_int c)		const	;
    template <class OP, class S, class B>
    SparseMatrix&	apply(u_int i, u_int j,
			      OP op, const Vector<S, B>& v)		;
    template <class OP, class S, class B, class R>
    SparseMatrix&	apply(u_int i, u_int j,
			      OP op, const Matrix<S, B, R>& M)		;

  // �A���ꎟ������
    Vector<T>		solve(const Vector<T>& b)		const	;

  // ���o��
    template <class S, bool SYMM> friend std::ostream&
			operator <<(std::ostream& out,
				    const SparseMatrix<S, SYMM>& A)	;
    template <class S, bool SYMM> friend std::istream&
			operator >>(std::istream& in,
				    SparseMatrix<S, SYMM>& A)		;

    friend class SparseMatrix<T, !SYM>;
    
  private:
    template <class OP>
    SparseMatrix	binary_op(const SparseMatrix& A, OP op)	const	;
    template <bool SYM2>
    bool		inner_product(const SparseMatrix<T, SYM2>& S,
				      u_int i, u_int j, T& val)	const	;
    int			index(u_int i, u_int j)			const	;
    static int		pardiso_precision()				;

  private:
    u_int		_ncol;		//!< ��̐�
    std::vector<int>	_rowIndex;	//!< �e�s�̐擪�����̒ʂ��ԍ�(1�x�[�X)
    std::vector<int>	_columns;	//!< �e�����̗�ԍ�(1�x�[�X)
    std::vector<T>	_values;	//!< �e�����̒l
};

/*
 * ----------------------- �\������ -----------------------------
 */
//! ���������J�n����D
template <class T, bool SYM> inline void
SparseMatrix<T, SYM>::beginInit()
{
    _ncol = 0;
    _rowIndex.clear();
    _columns.clear();
    _values.clear();
}

//! �s�̐擪�ʒu���Z�b�g����D
template <class T, bool SYM> inline void
SparseMatrix<T, SYM>::setRow()
{
  // 1�O�̍s�̗�ԍ����\�[�g
    if (!_rowIndex.empty())
	std::sort(_columns.begin() + _rowIndex.back() - 1, _columns.end());

  // ���̍s�̐擪�����̒ʂ��ԍ����Z�b�g(1�x�[�X)
    _rowIndex.push_back(nelements() + 1);
}

//! �����̗�ԍ����Z�b�g����D
/*!
  \param col	��ԍ�
*/
template <class T, bool SYM> inline void
SparseMatrix<T, SYM>::setCol(u_int col)
{
    u_int	row = _rowIndex.size();	// ���݂܂łɃZ�b�g���ꂽ�s��
    if (row == 0)
	throw std::runtime_error("TU::SparseMatrix<T, SYM>::setCol(): _rowIndex is not set!");
    if (SYM)
    {
	if (col < --row)		// �^����ꂽ��ԍ������݂̍s�ԍ��Ɣ�r
	    throw std::invalid_argument("TU::SparseMatrix<T, SYM>::setCol(): column index must not be less than row index!");
    }

    _columns.push_back(col + 1);	// ���̐����̗�ԍ����Z�b�g(1�x�[�X)
    if (col + 1 > _ncol)
	_ncol = col + 1;		// �񐔂��X�V
}

//! �����̗�ԍ��Ƃ��̒l���Z�b�g����D
/*!
  \param col	��ԍ�
  \param val	�l
*/
template <class T, bool SYM> inline void
SparseMatrix<T, SYM>::setCol(u_int col, value_type val)
{
    setCol(col);
    _values.push_back(val);
}
    
//! ���O�̍s�Ɠ����ʒu�ɔ�됬�������悤�ȍs���Z�b�g����D
template <class T, bool SYM> inline void
SparseMatrix<T, SYM>::copyRow()
{
    u_int	row = _rowIndex.size();	// ���݂܂łɃZ�b�g���ꂽ�s��
    if (row == 0)
	throw std::runtime_error("TU::SparseMatrix<T, SYM>::copyRow(): no previous rows!");

  // �s�̐擪�����̒ʂ��ԍ����Z�b�g����Drow�����݂̍s�ԍ��ƂȂ�D
    setRow();

  // ���O�̍s��(�Ώ̍s��̏ꍇ��2�Ԗڈȍ~��)�����̗�ԍ������݂̍s�ɃR�s�[����D
    for (u_int n  = _rowIndex[row - 1] - (SYM ? 0 : 1),
	       ne = _rowIndex[row] - 1; n < ne; ++n)
	_columns.push_back(_columns[n]);
}

//! ����������������D
template <class T, bool SYM> inline void
SparseMatrix<T, SYM>::endInit()
{
    setRow();				// �_�~�[�̍s���Z�b�g

    if (SYM && (nrow() != ncol()))
	throw std::runtime_error("SparseMatrix<T, true>::endInit(): the numbers of rows and columns must be equal!");
    
    if (_values.empty())
    {
	_values.resize(nelements());	// �������i�[����̈���m��
	operator =(T(0));
    }
    else if (_values.size() != _columns.size())
	throw std::runtime_error("SparseMatrix<T, SYM>::endInit(): the number of values is inconsistent with the number of column indices!");
}

/*
 * ----------------------- ��{��� ---------------------------------
 */
//! �s��̎������Ȃ킿�s����Ԃ��D
/*!
  \return	�s��̎���(=�s��)
*/
template <class T, bool SYM> inline u_int
SparseMatrix<T, SYM>::dim() const
{
    return nrow();
}
    
//! �s��̍s����Ԃ��D
/*!
  \return	�s��̍s��
*/
template <class T, bool SYM> inline u_int
SparseMatrix<T, SYM>::nrow() const
{
    return _rowIndex.size() - 1;
}
    
//! �s��̗񐔂�Ԃ��D
/*!
  \return	�s��̗�
*/
template <class T, bool SYM> inline u_int
SparseMatrix<T, SYM>::ncol() const
{
    return _ncol;
}
    
//! �s��̔�됬������Ԃ��D
/*!
  \return	�s��̔�됬����
*/
template <class T, bool SYM> inline u_int
SparseMatrix<T, SYM>::nelements() const
{
    return _columns.size();
}
    
//! �w�肳�ꂽ�s�̐�������Ԃ��D
/*!
  \param i	�s�ԍ�
  \return	��i�s�̐�����
*/
template <class T, bool SYM> inline u_int
SparseMatrix<T, SYM>::nelements(u_int i) const
{
    return _rowIndex[i+1] - _rowIndex[i];
}
    
//! �w�肳�ꂽ�s�Ɨ�ɑΉ����鐬����Ԃ��D
/*!
  \param i			�s�ԍ�
  \param j			��ԍ�
  \return			(i, j)����
*/
template <class T, bool SYM> inline T
SparseMatrix<T, SYM>::operator ()(u_int i, u_int j) const
{
    const int	n = index(i, j);
    return (n >= 0 ? _values[n] : T(0));
}

//! �w�肳�ꂽ�s�Ɨ�ɑΉ����鐬����Ԃ��D
/*!
  \param i			�s�ԍ�
  \param j			��ԍ�
  \return			(i, j)����
  \throw std::out_of_range	���̍s��(i, j)�����������Ȃ��ꍇ�ɑ��o
*/
template <class T, bool SYM> inline T&
SparseMatrix<T, SYM>::operator ()(u_int i, u_int j)
{
    const int	n = index(i, j);
    if (n < 0)
	throw std::out_of_range("TU::SparseMatrix<T, SYM>::operator (): non-existent element!");
    return _values[n];
}
    
/*
 * ----------------------- ��{���Z ---------------------------------
 */
//! ���ׂĂ�0�łȂ������ɒ萔��������D
/*!
  \param c	�������萔
  \return	���̑a�s��
*/
template <class T, bool SYM> inline SparseMatrix<T, SYM>&
SparseMatrix<T, SYM>::operator =(value_type c)
{
    std::fill(_values.begin(), _values.end(), c);
    return *this;
}
    
//! ���̑a�s��ɒ萔���|����D
/*!
  \param c	�|����萔
  \return	���̑a�s��
*/
template <class T, bool SYM> SparseMatrix<T, SYM>&
SparseMatrix<T, SYM>::operator *=(value_type c)
{
    std::transform(_values.begin(), _values.end(), _values.begin(),
		   std::bind2nd(std::multiplies<T>(), c));
    return *this;
}
    
//! ���̑a�s���萔�Ŋ���D
/*!
  \param c	�|����萔
  \return	���̑a�s��
*/
template <class T, bool SYM> SparseMatrix<T, SYM>&
SparseMatrix<T, SYM>::operator /=(value_type c)
{
    std::transform(_values.begin(), _values.end(), _values.begin(),
		   std::bind2nd(std::divides<T>(), c));
    return *this;
}
    
//! ���̑a�s��ɑ��̑a�s��𑫂��D
/*!
  2�̑a�s��͓���̍\���������˂΂Ȃ�Ȃ��D
  \param A			�����a�s��
  \return			���̑a�s��
  \throw std::invalid_argument	2�̑a�s��̍\������v���Ȃ��ꍇ�ɑ��o
*/
template <class T, bool SYM> SparseMatrix<T, SYM>&
SparseMatrix<T, SYM>::operator +=(const SparseMatrix& A)
{
    if (nrow()	    != A.nrow()						||
	nelements() != A.nelements()					||
	!equal(_rowIndex.begin(), _rowIndex.end(), A._rowIndex.begin())	||
	!equal(_columns.begin(),  _columns.end(),  A._columns.begin()))
	throw std::invalid_argument("TU::SparseMatrix<T, SYM>::operator +=(): structure mismatch!");
    std::transform(_values.begin(), _values.end(),
		   A._values.begin(), _values.begin(), std::plus<T>());
    return *this;
}
    
//! ���̑a�s�񂩂瑼�̑a�s��������D
/*!
  2�̑a�s��͓���̍\���������˂΂Ȃ�Ȃ��D
  \param A			�����a�s��
  \return			���̑a�s��
  \throw std::invalid_argument	2�̑a�s��̍\������v���Ȃ��ꍇ�ɑ��o
*/
template <class T, bool SYM> SparseMatrix<T, SYM>&
SparseMatrix<T, SYM>::operator -=(const SparseMatrix& A)
{
    if (nrow()	    != A.nrow()						||
	nelements() != A.nelements()					||
	!equal(_rowIndex.begin(), _rowIndex.end(), A._rowIndex.begin())	||
	!equal(_columns.begin(),  _columns.end(),  A._columns.begin()))
	throw std::invalid_argument("TU::SparseMatrix<T, SYM>::operator -=(): structure mismatch!");
    std::transform(_values.begin(), _values.end(),
		   A._values.begin(), _values.begin(), std::minus<T>());
    return *this;
}

//! ���̑a�s��Ƒ��̑a�s��̘a���v�Z����D
/*!
  2�̑a�s��͓���̃T�C�Y�������˂΂Ȃ�Ȃ��D
  \param A	�����a�s��
  \return	2�̑a�s��̘a
*/
template <class T, bool SYM> inline SparseMatrix<T, SYM>
SparseMatrix<T, SYM>::operator +(const SparseMatrix& A) const
{
    return binary_op(A, std::plus<T>());
}
    
//! ���̑a�s��Ƒ��̑a�s��̍����v�Z����D
/*!
  2�̑a�s��͓���̃T�C�Y�������˂΂Ȃ�Ȃ��D
  \param A	�����a�s��
  \return	2�̑a�s��̍�
*/
template <class T, bool SYM> inline SparseMatrix<T, SYM>
SparseMatrix<T, SYM>::operator -(const SparseMatrix& A) const
{
    return binary_op(A, std::minus<T>());
}
    
//! ���̑a�s��ɉE����x�N�g�����|����D
/*!
  \param v	�|����x�N�g��
  \return	���ʂ��i�[�����x�N�g��
*/
template <class T, bool SYM> template <class S, class B> Vector<T>
SparseMatrix<T, SYM>::operator *(const Vector<S, B>& v) const
{
    if (v.dim() != ncol())
	throw std::invalid_argument("SparseMatrxi<T, SYM>::operator *(): dimension mismatch!");
    
    Vector<T>	a(nrow());
    for (u_int i = 0; i < nrow(); ++i)
    {
	if (SYM)
	{
	    for (u_int j = 0; j < i; ++j)
	    {
		const int	n = index(i, j);
		if (n >= 0)
		    a[i] += _values[n] * v[j];
	    }
	}
	
	for (u_int n = _rowIndex[i] - 1; n < _rowIndex[i+1] - 1; ++n)
	    a[i] += _values[n] * v[_columns[n] - 1];
    }

    return a;
}

//! ���̑a�s��ɉE���玩�g�̓]�u���|�����s���Ԃ��D
/*!
  \return	���ʂ��i�[�����a�Ώ̍s��
*/
template <class T, bool SYM> SparseMatrix<T, true>
SparseMatrix<T, SYM>::compose() const
{
    SparseMatrix<T, true>	A;	// ���ʂ��i�[����a�Ώ̍s��

    A.beginInit();
    for (u_int i = 0; i < nrow(); ++i)
    {
	A.setRow();
	
	for (u_int j = i; j < nrow(); ++j)
	{
	    T	val;
	    if (inner_product(*this, i, j, val))
		A.setCol(j, val);
	}
    }
    A.endInit();
    
    return A;
}
    
//! ���̑a�s��ɉE����^����ꂽ�a�Ώ̍s��Ǝ��g�̓]�u���|�����s���Ԃ��D
/*!
  \return	���ʂ��i�[�����a�Ώ̍s��
*/
template <class T, bool SYM> SparseMatrix<T, true>
SparseMatrix<T, SYM>::compose(const SparseMatrix<T, true>& W) const
{
    if (ncol() != W.nrow())
	throw std::runtime_error("TU::SparseMatrix<T, SYM>::compose(): mismatched dimension!");

    SparseMatrix<T, false>	A;
    A.beginInit();
    for (u_int i = 0; i < nrow(); ++i)
    {
	A.setRow();

	for (u_int j = 0; j < W.nrow(); ++j)
	{
	    T	val;
	    if (inner_product(W, i, j, val))
		A.setCol(j, val);
	}
    }
    A.endInit();

    SparseMatrix<T, true>	B;
    B.beginInit();
    for (u_int i = 0; i < A.nrow(); ++i)
    {
	B.setRow();

	for (u_int j = i; j < nrow(); ++j)
	{
	    T	val;
	    if (A.inner_product(*this, i, j, val))
		B.setCol(j, val);
	}
    }
    B.endInit();

    return B;
}

/*
 * ----------------------- �u���b�N���Z -----------------------------
 */
//! �a�s��̍s���̖��ȕ������x�N�g���Ƃ��Ď��o���D
/*!
  ���̑a�s�񂪔�Ώ̂܂��� i <= j �̏ꍇ�́C(i, j)�������N�_�Ƃ��čs������
  d �̘A���������������o���D�Ώ̂��� i > j �̏ꍇ�́C(j, i)�������N�_��
  ���ė������ d �̘A���������������o���D
  \param i	�N�_�̍s�ԍ�
  \param j	�N�_�̗�ԍ�
  \param d	���o��������
  \return	���o������������ׂ��x�N�g��
*/
template <class T, bool SYM> Vector<T>
SparseMatrix<T, SYM>::operator ()(u_int i, u_int j, u_int d) const
{
    Vector<T>	v(d);
    const T*	p = &(*this)(i, j);
    u_int	dj = 0;
    if (SYM)
    {
	for (u_int je = std::min(i, j + v.dim()); j + dj < je; ++dj)
	{
	    v[dj] = *p;
	    p += (nelements(j + dj) - 1);
	}
    }
    for (; dj < v.dim(); ++dj)
	v[dj] = *p++;
    
    return v;
}

//! �a�s�񒆂̖��ȃu���b�N���s��Ƃ��Ď��o���D
/*!
  ���̑a�s�񂪔�Ώ̂܂��� i <= j �̏ꍇ�́C(i, j)�������N�_�Ƃ���
  �A������ r * c ���������o���D�Ώ̂��� i > j �̏ꍇ�́C(j, i)������
  �N�_�Ƃ���A������ c * r ���������o���D
  \param i	�N�_�̍s�ԍ�
  \param j	�N�_�̗�ԍ�
  \param r	���o���s��
  \param c	���o����
  \return	���o������������ׂ��s��
*/
template <class T, bool SYM> Matrix<T>
SparseMatrix<T, SYM>::operator ()(u_int i, u_int j,
				  u_int r, u_int c) const
{
    Matrix<T>	M(r, c);
    for (u_int di = 0; di < M.nrow(); ++di)
	M[di] = (*this)(i + di, j, M.ncol());

    return M;
}

//! �^����ꂽ�x�N�g���̊e�������w�肳�ꂽ�������N�_�Ƃ��ēK�p����D
/*!
  ���̑a�s�񂪔�Ώ̂܂��� i <= j �̏ꍇ�́C(i, j)�������N�_�Ƃ���
  �s�����ɓK�p����D�Ώ̂��� i > j �̏ꍇ�́C(j, i)�������N�_�Ƃ���
  ������ɓK�p����D
  \param i	�N�_�̍s�ԍ�
  \param j	�N�_�̗�ԍ�
  \param f	T�^�CS�^�̈������Ƃ�T�^�̒l��Ԃ�2�����Z�q
  \param v	���̊e������f�̑�2�����ƂȂ�x�N�g��
  \return	���̑a�Ώ̍s��
*/
template <class T, bool SYM> template <class OP, class S, class B>
SparseMatrix<T, SYM>&
SparseMatrix<T, SYM>::apply(u_int i, u_int j, OP op, const Vector<S, B>& v)
{
    T*		p = &(*this)(i, j);
    u_int	dj = 0;
    if (SYM)
    {
	for (u_int je = std::min(i, j + v.dim()); j + dj < je; ++dj)
	{
	    *p = op(*p, v[dj]);
	    p += (nelements(j + dj) - 1);
	}
    }
    for (; dj < v.dim(); ++dj)
    {
	*p = op(*p, v[dj]);
	++p;
    }
    
    return *this;
}
    
//! �^����ꂽ�s��̊e�������w�肳�ꂽ�������N�_�Ƃ��ēK�p����D
/*!
  ���̑a�s�񂪔�Ώ̂܂��� i <= j �̏ꍇ�́C(i, j)�������N�_�Ƃ���
  �s�D�揇�ɓK�p����D�Ώ̂��� i > j �̏ꍇ�́C(j, i)�������N�_�Ƃ���
  ��D�揇�ɓK�p����D
  \param i	�N�_�̍s�ԍ�
  \param j	�N�_�̗�ԍ�
  \param f	T�^�CS�^�̈������Ƃ�T�^�̒l��Ԃ�2�����Z�q
  \param M	���̊e������f�̑�2�����ƂȂ�s��
  \return	���̑a�Ώ̍s��
*/
template <class T, bool SYM> template <class OP, class S, class B, class R>
SparseMatrix<T, SYM>&
SparseMatrix<T, SYM>::apply(u_int i, u_int j, OP op, const Matrix<S, B, R>& M)
{
    const Matrix<S, B, R>&	MM = (i > j ? std::swap(i, j), M.trns() : M);
    
    for (u_int di = 0; di < MM.nrow(); ++di)
    {
	const u_int	dj = (SYM ? std::max(i + di, j) - j : 0);
	apply(i + di, j + dj, op, MM[di](dj, MM.ncol() - dj));
    }
    
    return *this;
}
    
/*
 * ----------------------- �A���ꎟ������ -----------------------------
 */
//! ���̍s����W���Ƃ���A���ꎟ�������������D
/*!
  MKL direct sparse solver�ɂ����
  \f$\TUvec{A}{}\TUvec{x}{} = \TUvec{b}{}\f$�������D
  \param b	�x�N�g��
  \return	���x�N�g��
*/
template <class T, bool SYM> Vector<T>
SparseMatrix<T, SYM>::solve(const Vector<T>& b) const
{
    if (nrow() != ncol())
	throw std::runtime_error("TU::SparseMatrix<T, SYM>::solve(): not a square matrix!");
    if (b.dim() != ncol())
	throw std::runtime_error("TU::SparseMatrix<T, SYM>::solve(): input vector with invalid dimension!");

  // pardiso �̊e��p�����[�^��ݒ肷��D
    _MKL_DSS_HANDLE_t	pt[64];		// pardiso�̓����������ւ̃|�C���^
    for (int i = 0; i < 64; ++i)
	pt[i] = 0;
    _INTEGER_t		iparm[64];
    for (int i = 0; i < 64; ++i)
	iparm[i] = 0;
    iparm[0]  =  1;			// iparm[1-] �Ƀf�t�H���g�l�ȊO���w��
    iparm[1]  =  2;			// nested dissection algorithm
    iparm[9]  =  8;			// eps = 1.0e-8
    iparm[17] = -1;			// �����̔�됬���������|�[�g
    iparm[20] =  1;			// Bunch and Kaufman pivoting
    iparm[27] = pardiso_precision();	// float �܂��� double ���w��
    _INTEGER_t		maxfct = 1;	// ���̕�����ێ�����ׂ��s��̐�
    _INTEGER_t		mnum   = 1;	// ���Ԗڂ̍s��ɂ��ĉ��������w��
    _INTEGER_t		mtype  = (SYM ? -2 : 11);	// ���Ώ́^����Ώ̍s��
    _INTEGER_t		phase  = 13;	// �s��̉�͂��甽���ɂ��ז����܂�
    _INTEGER_t		n      = nrow();// �A���ꎟ�������̎���
    _INTEGER_t		nrhs   = 1;	// Ax = b �ɂ����� b �̗�
    _INTEGER_t		msglvl = 0;	// ���b�Z�[�W���o�͂��Ȃ�
    _INTEGER_t		error  = 0;	// �G���[�R�[�h
    Array<_INTEGER_t>	perm(n);	// permutation�x�N�g��
    Vector<T>		x(n);		// ���x�N�g��

  // �A���ꎟ�������������D
    PARDISO(&pt[0], &maxfct, &mnum, &mtype, &phase, &n, (void*)&_values[0],
	    (_INTEGER_t*)&_rowIndex[0], (_INTEGER_t*)&_columns[0],
	    &perm[0], &nrhs, iparm, &msglvl, (void*)&b[0], &x[0], &error);
    if (error != 0)
	throw std::runtime_error("TU::SparseMatrix<T, SYM>::solve(): PARDISO failed!");

  // pardiso ���Ŏg�p�����S���������������D
    phase = -1;
    PARDISO(&pt[0], &maxfct, &mnum, &mtype, &phase, &n, (void*)&_values[0],
	    (_INTEGER_t*)&_rowIndex[0], (_INTEGER_t*)&_columns[0],
	    &perm[0], &nrhs, iparm, &msglvl, (void*)&b[0], &x[0], &error);
    if (error != 0)
	throw std::runtime_error("TU::SparseMatrix<T, SYM>::solve(): PARDISO failed to release memory!");

    return x;
}

/*
 * ----------------------- private members -----------------------------
 */
//! �w�肳�ꂽ�s�Ɨ�ɑΉ����� #_values ��index��Ԃ��D
/*!
  #_rowIndex �Ɠ��l��1�x�[�X��index��Ԃ��̂ŁC���ۂ� #_values ��
  �A�N�Z�X����ۂɂ͕Ԃ��ꂽ�l����1�������K�v������D
  \param i	�s�ԍ�
  \param j	��ԍ�
  \return	(i, j)���������݂���΂���index(0�x�[�X), ���݂��Ȃ����-1
*/
template <class T, bool SYM> inline int
SparseMatrix<T, SYM>::index(u_int i, u_int j) const
{
    if (SYM && i > j)
	std::swap(i, j);

    ++j;				// ��ԍ���1�x�[�X�ɒ����D

  // �w�肳�ꂽ��ԍ��ɑΉ����鐬�������̍s�ɂ��邩2���@�ɂ���Ē��ׂ�D
    for (u_int low = _rowIndex[i]-1, high = _rowIndex[i+1]-1; low != high; )
    {
	u_int	mid = (low + high) / 2;

	if (j < _columns[mid])
	    high = mid;
	else if (j > _columns[mid])
	    low = mid + 1;
	else
	    return mid;
    }

    return -1;				// �݂���Ȃ����-1��Ԃ��D
}

template<> inline int
SparseMatrix<float,  false>::pardiso_precision()	{return 1;}
template<> inline int
SparseMatrix<float,  true> ::pardiso_precision()	{return 1;}
template<> inline int
SparseMatrix<double, false>::pardiso_precision()	{return 0;}
template<> inline int
SparseMatrix<double, true> ::pardiso_precision()	{return 0;}

/************************************************************************
*  global functions							*
************************************************************************/
//! ���̑a�s��Ƒ��̑a�s��̊ԂŐ�������2�����Z���s���D
/*!
  \param A			��������̑a�s��
  \return			2�̑a�s��Ԃ̐�������2�����Z�œ�����a�s��
  \throw std::invalid_argument	2�̑a�s��̃T�C�Y����v���Ȃ��ꍇ�ɑ��o
*/
template <class T, bool SYM> template <class OP> SparseMatrix<T, SYM>
SparseMatrix<T, SYM>::binary_op(const SparseMatrix& A, OP op) const
{
    if ((nrow() != A.nrow()) || (ncol() != A.ncol()))
	throw std::invalid_argument("SparseMatrix<T, SYM>::binary_op(): two matrices must have equal sizes!");

    SparseMatrix	S;

    S.beginInit();
    for (u_int i = 0; i < nrow(); ++i)
    {
	S.setRow();

	for (u_int m = _rowIndex[i], n = A._rowIndex[i]; ; )
	{
	    const u_int	j = (m <   _rowIndex[i+1] ?   _columns[m-1] - 1
						  :   ncol());
	    const u_int	k = (n < A._rowIndex[i+1] ? A._columns[n-1] - 1
						  : A.ncol());
	    
	    if (j == k)
	    {
		if (j == ncol())
		    break;
		
		S.setCol(j, op(_values[m-1], A._values[n-1]));
		++m;
		++n;
	    }
	    else if (j < k)
	    {
		S.setCol(j, op(_values[m-1], T(0)));
		++m;
	    }
	    else
	    {
		S.setCol(k, op(T(0), A._values[n-1]));
		++n;
	    }
	}
    }
    S.endInit();

    return S;
}
    
//! ���̑a�s��Ɨ^����ꂽ�a�s�񂩂炻�ꂼ��1�s�����o���C�����̓��ς����߂�D
/*!
  \param S	����1�̑a�s��
  \param i	���̑a�s��̍s�ԍ�
  \param j	S �̍s�ԍ�
  \param val	���̑a�s��̑�i�s�� M �̑�j�s�̓��ς̒l���Ԃ����
  \return	���̑a�s��̑�i�s�� M �̑�j�s����ԍ������Ȃ��Ƃ�
		1���L�����true, �����łȂ���� false
*/
template <class T, bool SYM> template <bool SYM2> bool
SparseMatrix<T, SYM>::inner_product(const SparseMatrix<T, SYM2>& S,
				    u_int i, u_int j, T& val) const
{
    if (ncol() != S.ncol())
	throw std::invalid_argument("inner_product(): mismatched dimension!");
    
    bool	exist = false;
    val = T(0);

    if (SYM)
    {
	for (u_int col = 0; col < i; ++col)
	{
	    int	m, n;
	    if ((m = index(i, col)) >= 0 && (n = S.index(j, col)) >= 0)
	    {
		exist = true;
		val += _values[m] * S._values[n];
	    }
	}
    }

    for (u_int m = _rowIndex[i] - 1; m < _rowIndex[i+1] - 1; ++m)
    {
	const int	n = S.index(j, _columns[m] - 1);
	if (n >= 0)
	{
	    exist = true;
	    val += _values[m] * S._values[n];
	}
    }

    return exist;
}

//! �o�̓X�g���[���֑a�s��������o��(ASCII)�C����ɉ��s�R�[�h���o�͂���D
/*!
  \param out	�o�̓X�g���[��
  \param A	�����o���a�s��
  \return	out�Ŏw�肵���o�̓X�g���[��
*/
template <class S, bool SYMM> std::ostream&
operator <<(std::ostream& out, const SparseMatrix<S, SYMM>& A)
{
    using namespace	std;

    u_int	n = 0;
    for (u_int i = 0; i < A.nrow(); ++i)
    {
	if (SYMM)
	{
	    for (u_int j = 0; j < i; ++j)
		out << " *";
	}

	for (u_int j = (SYMM ? i : 0); j < A.ncol(); ++j)
	{
	    if ((n < A._rowIndex[i+1] - 1) && (j == A._columns[n] - 1))
		out << ' ' << A._values[n++];
	    else
		out << " (0)";
	}
	out << endl;
    }

    return out << endl;
}

}

#endif	// ! __TUSparseMatrixPP_h