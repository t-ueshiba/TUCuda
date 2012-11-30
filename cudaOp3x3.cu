/*
 *  $Id: cudaOp3x3.cu,v 1.7 2011-05-09 00:35:49 ueshiba Exp $
 */
#include "TU/CudaUtility.h"

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
op3x3_kernel(const S* in, T* out, u_int stride_i, u_int stride_o, OP op)
{
  // ���̃J�[�l���̓u���b�N���E�����̂��߂� blockDim.x == blockDim.y ������
    const int	blk = blockDim.x;	// u_int�ɂ���ƃ_���DCUDA�̃o�O�H
    int		xy  = (blockIdx.y * blk + threadIdx.y) * stride_i 
		    +  blockIdx.x * blk + threadIdx.x;	// ���݈ʒu
    int		x   = 1 + threadIdx.x;
    const int	y   = 1 + threadIdx.y;

  // ���摜�̃u���b�N��������т��̊O�g1��f�������L�������ɓ]��
    __shared__ S	in_s[BlockDim+2][BlockDim+2];
    in_s[y][x] = in[xy];				// ����

    if (threadIdx.y == 0)	// �u���b�N�̏�[?
    {
	const int	top = xy - stride_i;		// ���݈ʒu�̒���
	const int	bot = xy + blk * stride_i;	// ���݈ʒu�̉��[
	in_s[	   0][x] = in[top];			// ��g
	in_s[blk + 1][x] = in[bot];			// ���g

	const int	lft = xy + threadIdx.x * (stride_i - 1);
	in_s[x][      0] = in[lft -   1];		// ���g
	in_s[x][blk + 1] = in[lft + blk];		// �E�g

	if (threadIdx.x == 0)	// �u���b�N�̍����?
	{
	    if ((blockIdx.x != 0) || (blockIdx.y != 0))
	  	in_s[0][0] = in[top - 1];		// �����
	    if ((blockIdx.x != gridDim.x - 1) || (blockIdx.y != gridDim.y - 1))
		in_s[blk + 1][blk + 1] = in[bot + blk];	// �E����
	    in_s[0][blk + 1] = in[top + blk];		// �E���
	    in_s[blk + 1][0] = in[bot -   1];		// ������
	}
    }
    __syncthreads();

  // ���L�������ɕۑ��������摜�f�[�^���猻�݉�f�ɑ΂���t�B���^�o�͂��v�Z
    xy = (blockIdx.y * blk + threadIdx.y) * stride_o
       +  blockIdx.x * blk + threadIdx.x;
    --x;
    out[xy] = op(in_s[y - 1] + x, in_s[y] + x, in_s[y + 1] + x);
}

/************************************************************************
*  global functions							*
************************************************************************/
//! CUDA�ɂ����2�����z��ɑ΂���3x3�ߖT���Z���s���D
/*!
  \param in	����2�����z��
  \param out	�o��2�����z��
  \param op	3x3�ߖT���Z�q
*/
template <class S, class T, class OP> void
cudaOp3x3(const CudaArray2<S>& in, CudaArray2<T>& out, OP op)
{
    using namespace	std;
    
    if (in.nrow() < 3 || in.ncol() < 3)
	return;
    
    out.resize(in.nrow(), in.ncol());

  // �ŏ��ƍŌ�̍s�������� (out.nrow() - 2) x out.stride() �̔z��Ƃ��Ĉ���
    dim3	threads(BlockDim, BlockDim);
    dim3	blocks(out.stride()/threads.x, (out.nrow() - 2)/threads.y);
    op3x3_kernel<<<blocks, threads>>>(in[1].ptr(), out[1].ptr(),
				      in.stride(), out.stride(), op);

  // ����
    int	top = 1 + threads.y * blocks.y;
    threads.x = threads.y = out.nrow() - top - 1;
    if (threads.x == 0)
	return;
    blocks.x = out.stride() / threads.x;
    blocks.y = 1;
    op3x3_kernel<<<blocks, threads>>>(in[top].ptr(), out[top].ptr(),
				      in.stride(), out.stride(), op);

  // �E��
    if (threads.x * blocks.x == out.stride())
	return;
    int	lft = out.stride() - threads.x;
    blocks.x = 1;
    op3x3_kernel<<<blocks, threads>>>(in[top].ptr()  + lft,
				      out[top].ptr() + lft,
				      in.stride(), out.stride(), op);
}

template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  diffH3x3<u_char, float> op)					;
template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  diffH3x3<float, float> op)					;

template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  diffV3x3<u_char, float> op)					;
template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  diffV3x3<float, float> op)					;

template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  diffHH3x3<u_char, float> op)					;
template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  diffHH3x3<float, float> op)					;

template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  diffVV3x3<u_char, float> op)					;
template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  diffVV3x3<float, float> op)					;

template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  diffHV3x3<float, float> op)					;
template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  diffHV3x3<u_char, float> op)					;

template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  sobelH3x3<u_char, float> op)					;
template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  sobelH3x3<float, float> op)					;

template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  sobelV3x3<u_char, float> op)					;
template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  sobelV3x3<float, float> op)					;

template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  sobelAbs3x3<u_char, float> op)				;
template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  sobelAbs3x3<float, float> op)					;

template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  laplacian3x3<u_char, float> op)				;
template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  laplacian3x3<float, float> op)				;

template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  det3x3<float, float> op)					;
template void
cudaOp3x3(const CudaArray2<u_char>& in, CudaArray2<float>& out,
	  det3x3<u_char, float> op)					;

template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  maximal3x3<float> op)						;

template void
cudaOp3x3(const CudaArray2<float>& in, CudaArray2<float>& out,
	  minimal3x3<float> op)						;
}
