/*
 *  $Id: cudaSuppressNonExtrema3x3.cu,v 1.1 2011-04-28 07:59:04 ueshiba Exp $
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
template <class S, class T, class OP> static __global__ void
extrema3x3_kernel(const S* in, T* out,
		  u_int stride_i, u_int stride_o, OP op, T nulval)
{
  // ���̃J�[�l���̓u���b�N���E�����̂��߂� blockDim.x == blockDim.y ������
    const int	blk = blockDim.x;	// u_int�ɂ���ƃ_���DCUDA�̃o�O�H
    int		xy  = 2*(blockIdx.y*blockDim.y + threadIdx.y)*stride_i
		    + 2*blockIdx.x*blockDim.y + threadIdx.x;
    int		x_s = 1 +   threadIdx.x;
    int		y_s = 1 + 2*threadIdx.y;

  // (2*blockDim.x)x(2*blockDim.y) �̋�`�̈����͗p���L�������̈�ɃR�s�[
    __shared__ S	in_s[2*BlockDim + 2][2*BlockDim + 3];
    in_s[y_s    ][x_s	   ] = in[xy		     ];
    in_s[y_s    ][x_s + blk] = in[xy		+ blk];
    in_s[y_s + 1][x_s	   ] = in[xy + stride_i	     ];
    in_s[y_s + 1][x_s + blk] = in[xy + stride_i + blk];

  // 2x2�u���b�N�̊O�g����͗p���L�������̈�ɃR�s�[
    if (threadIdx.y == 0)	// �u���b�N�̏�[?
    {
	const int	blk2 = 2*blockDim.y;
	const int	top  = xy - stride_i;		// ���݈ʒu�̒���
	const int	bot  = xy + blk2 * stride_i;	// ���݈ʒu�̉��[
	in_s[0	     ][x_s	] = in[top	];	// ��g����
	in_s[0	     ][x_s + blk] = in[top + blk];	// ��g�E��
	in_s[1 + blk2][x_s	] = in[bot	];	// ���g����
	in_s[1 + blk2][x_s + blk] = in[bot + blk];	// ���g�E��

	int	lft = xy + threadIdx.x*(stride_i - 1);
	in_s[x_s      ][0	] = in[lft -	1];	// ���g�㔼
	in_s[x_s      ][1 + blk2] = in[lft + blk2];	// �E�g�㔼
	lft += blockDim.y * stride_i;
	in_s[x_s + blk][0	] = in[lft -	1];	// ���g����
	in_s[x_s + blk][1 + blk2] = in[lft + blk2];	// �E�g����

	if (threadIdx.x == 0)	// �u���b�N�̍����?
	{
	    if ((blockIdx.x != 0) || (blockIdx.y != 0))
	  	in_s[0][0] = in[top - 1];			// �����
	    if ((blockIdx.x != gridDim.x - 1) || (blockIdx.y != gridDim.y - 1))
		in_s[1 + blk2][1 + blk2] = in[bot + blk2];	// �E����
	    in_s[0][1 + blk2] = in[top + blk2];			// �E���
	    in_s[1 + blk2][0] = in[bot -    1];			// ������
	}
    }
    __syncthreads();

  // ���̃X���b�h�̏����Ώۂł���2x2��f�E�B���h�E�ɔ�ɒl��\���l���������ށD
    __shared__ T	out_s[2*BlockDim][2*BlockDim+1];
    x_s = 1 + 2*threadIdx.x;
    out_s[y_s - 1][x_s - 1] = out_s[y_s - 1][x_s]
			    = out_s[y_s][x_s - 1]
			    = out_s[y_s][x_s	] = nulval;

  // ����2x2�E�B���h�E���ōő�/�ŏ��ƂȂ��f�̍��W�����߂�D
    const int	i01 = (op(in_s[y_s    ][x_s], in_s[y_s	  ][x_s + 1]) ? 0 : 1);
    const int	i23 = (op(in_s[y_s + 1][x_s], in_s[y_s + 1][x_s + 1]) ? 2 : 3);
    const int	iex = (op(in_s[y_s    ][x_s + i01],
			  in_s[y_s + 1][x_s + (i23 & 0x1)]) ? i01 : i23);
    x_s += (iex & 0x1);
    y_s += (iex >> 1);
    
    const T	val = in_s[y_s][x_s];
    const int	dx  = (iex & 0x1 ? 1 : -1);
    const int	dy  = (iex & 0x2 ? 1 : -1);
    out_s[y_s-1][x_s-1] = (op(val, in_s[y_s + dy][x_s - dx]) &&
			   op(val, in_s[y_s + dy][x_s	  ]) &&
			   op(val, in_s[y_s + dy][x_s + dx]) &&
			   op(val, in_s[y_s     ][x_s + dx]) &&
			   op(val, in_s[y_s - dy][x_s + dx]) ? val : nulval);
    __syncthreads();

  // (2*blockDim.x)x(2*blockDim.y) �̋�`�̈�ɏo�͗p���L�������̈���R�s�[�D
    x_s =   threadIdx.x;
    y_s = 2*threadIdx.y;
    xy  = (2*blockIdx.y*blockDim.y + y_s)*stride_o
	+  2*blockIdx.x*blockDim.y + x_s;
    out[xy		   ] = out_s[y_s    ][x_s      ];
    out[xy	      + blk] = out_s[y_s    ][x_s + blk];
    out[xy + stride_o	   ] = out_s[y_s + 1][x_s      ];
    out[xy + stride_o + blk] = out_s[y_s + 1][x_s + blk];
}
    
/************************************************************************
*  global functions							*
************************************************************************/
//! CUDA�ɂ����2�����z��ɑ΂���3x3��ɒl�}���������s���D
/*!
  \param in	����2�����z��
  \param out	�o��2�����z��
*/
template <class S, class T, class OP> void
cudaSuppressNonExtrema3x3(const CudaArray2<S>& in,
				CudaArray2<T>& out, OP op, T nulval)
{
    using namespace	std;
    
    if (in.nrow() < 3 || in.ncol() < 3)
	return;
    
    out.resize(in.nrow(), in.ncol());

  // �ŏ��ƍŌ�̍s�������� (out.nrow() - 2) x out.stride() �̔z��Ƃ��Ĉ���
    dim3	threads(BlockDim, BlockDim);
    dim3	blocks(out.stride()     / (2*threads.x),
		       (out.nrow() - 2) / (2*threads.y));
    extrema3x3_kernel<<<blocks, threads>>>((const S*)in[1], (T*)out[1],
					   in.stride(), out.stride(),
					   op, nulval);

  // ����
    int	ys = 1 + 2*threads.y * blocks.y;
    threads.x = threads.y = (out.nrow() - ys - 1) / 2;
    if (threads.x == 0)
	return;
    blocks.x = out.stride() / (2*threads.x);
    blocks.y = 1;
    extrema3x3_kernel<<<blocks, threads>>>((const S*)in[ys], (T*)out[ys],
					   in.stride(), out.stride(),
					   op, nulval);

  // �E��
    if (2*threads.x * blocks.x == out.stride())
	return;
    int	xs = out.stride() - 2*threads.x;
    blocks.x = 1;
    extrema3x3_kernel<<<blocks, threads>>>((const S*)in[ys]  + xs,
					   (	  T*)out[ys] + xs,
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
