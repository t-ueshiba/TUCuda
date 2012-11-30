/*
 *  $Id: cudaSuppressNonExtrema3x3.cu,v 1.2 2011-05-09 00:35:49 ueshiba Exp $
 */
#include "TU/CudaUtility.h"
#include <thrust/functional.h>

namespace TU
{
/************************************************************************
*  global constatnt variables						*
************************************************************************/
static const u_int	BlockDim = 16;		// �u���b�N�T�C�Y�̏����l
    
/************************************************************************
*  device functions							*
************************************************************************/
template <class T, class OP> static __global__ void
extrema3x3_kernel(const T* in, T* out,
		  u_int stride_i, u_int stride_o, OP op, T nulval)
{
  // ���̃J�[�l���̓u���b�N���E�����̂��߂� blockDim.x == blockDim.y ������
    const int	blk = blockDim.x;	// u_int�ɂ���ƃ_���DCUDA�̃o�O�H
    int		xy  = 2*(blockIdx.y*blockDim.y + threadIdx.y)*stride_i
		    + 2* blockIdx.x*blockDim.x + threadIdx.x;
    int		x   = 1 +   threadIdx.x;
    const int	y   = 1 + 2*threadIdx.y;

  // (2*blockDim.x)x(2*blockDim.y) �̋�`�̈�ɋ��L�������̈�ɃR�s�[
    __shared__ T	buf[2*BlockDim + 2][2*BlockDim + 3];
    buf[y    ][x      ] = in[xy			];
    buf[y    ][x + blk] = in[xy		   + blk];
    buf[y + 1][x      ] = in[xy + stride_i	];
    buf[y + 1][x + blk] = in[xy + stride_i + blk];

  // 2x2�u���b�N�̊O�g�����L�������̈�ɃR�s�[
    if (threadIdx.y == 0)	// �u���b�N�̏�[?
    {
	const int	blk2 = 2*blockDim.y;
	const int	top  = xy - stride_i;		// ���݈ʒu�̒���
	const int	bot  = xy + blk2 * stride_i;	// ���݈ʒu�̉��[
	buf[0	    ][x      ] = in[top      ];		// ��g����
	buf[0	    ][x + blk] = in[top + blk];		// ��g�E��
	buf[1 + blk2][x      ] = in[bot      ];		// ���g����
	buf[1 + blk2][x + blk] = in[bot + blk];		// ���g�E��

	int	lft = xy + threadIdx.x*(stride_i - 1);
	buf[x      ][0       ] = in[lft -    1];	// ���g�㔼
	buf[x      ][1 + blk2] = in[lft + blk2];	// �E�g�㔼
	lft += blockDim.y * stride_i;
	buf[x + blk][0       ] = in[lft -    1];	// ���g����
	buf[x + blk][1 + blk2] = in[lft + blk2];	// �E�g����

	if (threadIdx.x == 0)	// �u���b�N�̍����?
	{
	    if ((blockIdx.x != 0) || (blockIdx.y != 0))
	  	buf[0][0] = in[top - 1];			// �����
	    if ((blockIdx.x != gridDim.x - 1) || (blockIdx.y != gridDim.y - 1))
		buf[1 + blk2][1 + blk2] = in[bot + blk2];	// �E����
	    buf[0][1 + blk2] = in[top + blk2];			// �E���
	    buf[1 + blk2][0] = in[bot -    1];			// ������
	}
    }
    __syncthreads();

  // ���̃X���b�h�̏����Ώۂł���2x2�E�B���h�E���ōő�/�ŏ��ƂȂ��f�̍��W�����߂�D
    x = 1 + 2*threadIdx.x;
  //const int	i01 = (op(buf[y    ][x], buf[y	  ][x + 1]) ? 0 : 1);
  //const int	i23 = (op(buf[y + 1][x], buf[y + 1][x + 1]) ? 2 : 3);
    const int	i01 = op(buf[y    ][x + 1], buf[y    ][x]);
    const int	i23 = op(buf[y + 1][x + 1], buf[y + 1][x]) + 2;
    const int	iex = (op(buf[y][x + i01], buf[y + 1][x + (i23 & 0x1)]) ?
		       i01 : i23);
    const int	xx  = x + (iex & 0x1);		// �ő�/�ŏ��_��x���W
    const int	yy  = y + (iex >> 1);		// �ő�/�ŏ��_��y���W

  // �ő�/�ŏ��ƂȂ�����f���C�c��5�̋ߖT�_�����傫��/�����������ׂ�D
  //const int	dx  = (iex & 0x1 ? 1 : -1);
  //const int	dy  = (iex & 0x2 ? 1 : -1);
    const int	dx  = ((iex & 0x1) << 1) - 1;
    const int	dy  = (iex & 0x2) - 1;
    T		val = buf[yy][xx];
    val = (op(val, buf[yy + dy][xx - dx]) &
	   op(val, buf[yy + dy][xx     ]) &
	   op(val, buf[yy + dy][xx + dx]) &
	   op(val, buf[yy     ][xx + dx]) &
	   op(val, buf[yy - dy][xx + dx]) ? val : nulval);
    __syncthreads();

  // ����2x2��f�E�B���h�E�ɑΉ����鋤�L�������̈�ɏo�͒l���������ށD
    buf[y    ][x    ] = nulval;		// ��ɒl
    buf[y    ][x + 1] = nulval;		// ��ɒl
    buf[y + 1][x    ] = nulval;		// ��ɒl
    buf[y + 1][x + 1] = nulval;		// ��ɒl
    buf[yy   ][xx   ] = val;		// �ɒl�܂��͔�ɒl
    __syncthreads();
    
  // (2*blockDim.x)x(2*blockDim.y) �̋�`�̈�ɋ��L�������̈���R�s�[�D
    x = 1 + threadIdx.x;
    xy  = 2*(blockIdx.y*blockDim.y + threadIdx.y)*stride_o
	+ 2* blockIdx.x*blockDim.x + threadIdx.x;
    out[xy		   ] = buf[y    ][x      ];
    out[xy	      + blk] = buf[y    ][x + blk];
    out[xy + stride_o	   ] = buf[y + 1][x      ];
    out[xy + stride_o + blk] = buf[y + 1][x + blk];
}
    
/************************************************************************
*  global functions							*
************************************************************************/
//! CUDA�ɂ����2�����z��ɑ΂���3x3��ɒl�}���������s���D
/*!
  \param in	����2�����z��
  \param out	�o��2�����z��
  \param op	�ɑ�l�����o����Ƃ��� thrust::greater<T> ���C
		�ɏ��l�����o����Ƃ��� thrust::less<T> ��^����
  \param nulval	��ɒl���Ƃ��f�Ɋ��蓖�Ă�l
*/
template <class T, class OP> void
cudaSuppressNonExtrema3x3(const CudaArray2<T>& in,
				CudaArray2<T>& out, OP op, T nulval)
{
    if (in.nrow() < 3 || in.ncol() < 3)
	return;
    
    out.resize(in.nrow(), in.ncol());

  // �ŏ��ƍŌ�̍s�������� (out.nrow() - 2) x out.stride() �̔z��Ƃ��Ĉ���
    dim3	threads(BlockDim, BlockDim);
    dim3	blocks(out.stride()     / (2*threads.x),
		       (out.nrow() - 2) / (2*threads.y));
    extrema3x3_kernel<<<blocks, threads>>>(in[1].ptr(), out[1].ptr(),
					   in.stride(), out.stride(),
					   op, nulval);

  // ����
    int	ys = 1 + 2*threads.y * blocks.y;
    threads.x = threads.y = (out.nrow() - ys - 1) / 2;
    if (threads.x == 0)
	return;
    blocks.x = out.stride() / (2*threads.x);
    blocks.y = 1;
    extrema3x3_kernel<<<blocks, threads>>>(in[ys].ptr(), out[ys].ptr(),
					   in.stride(), out.stride(),
					   op, nulval);

  // �E��
    if (2*threads.x * blocks.x == out.stride())
	return;
    int	xs = out.stride() - 2*threads.x;
    blocks.x = 1;
    extrema3x3_kernel<<<blocks, threads>>>(in[ys].ptr()  + xs,
					   out[ys].ptr() + xs,
					   in.stride(), out.stride(),
					   op, nulval);
}

template  void
cudaSuppressNonExtrema3x3(const CudaArray2<u_char>& in,
			  CudaArray2<u_char>& out,
			  thrust::greater<u_char> op, u_char nulval);
template  void
cudaSuppressNonExtrema3x3(const CudaArray2<float>& in,
			  CudaArray2<float>& out,
			  thrust::greater<float> op, float nulval);
template  void
cudaSuppressNonExtrema3x3(const CudaArray2<u_char>& in,
			  CudaArray2<u_char>& out,
			  thrust::less<u_char> op, u_char nulval);
template  void
cudaSuppressNonExtrema3x3(const CudaArray2<float>& in,
			  CudaArray2<float>& out,
			  thrust::less<float> op, float nulval);
}
