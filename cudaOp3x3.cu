/*
 *  $Id: cudaOp3x3.cu,v 1.5 2011-04-26 04:53:39 ueshiba Exp $
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
    int	blk = blockDim.x;		// u_int�ɂ���ƃ_���DCUDA�̃o�O�H
    int	lft = (blockIdx.y * blk + threadIdx.y) * stride_i 
	    +  blockIdx.x * blk,			// ���݈ʒu���猩�č��[
	xy  = lft + threadIdx.x;			// ���݈ʒu
    int	x_s = threadIdx.x + 1,
	y_s = threadIdx.y + 1;

  // ���摜�̃u���b�N��������т��̊O�g1��f�������L�������ɓ]��
    __shared__ S	in_s[BlockDim+2][BlockDim+2];

    in_s[y_s][x_s] = in[xy];				// ����

    if (threadIdx.y == 0)	// �u���b�N�̏�[?
    {

	int	top = xy - stride_i;			// ���݈ʒu�̒���
	int	bot = xy + blk * stride_i;		// ���݈ʒu���猩�ĉ��[
	in_s[	   0][x_s] = in[top];			// ��g
	in_s[blk + 1][x_s] = in[bot];			// ���g

	lft += threadIdx.x * stride_i;
	in_s[x_s][	0] = in[lft - 1];		// ���g
	in_s[x_s][blk + 1] = in[lft + blk];		// �E�g

	if (threadIdx.x == 0)	// �u���b�N�̍����?
	{
	    if ((blockIdx.x != 0) || (blockIdx.y != 0))
	  	in_s[0][0] = in[top - 1];		// �����
	    if ((blockIdx.x != gridDim.x - 1) || (blockIdx.y != gridDim.y - 1))
		in_s[blk + 1][blk + 1] = in[bot + blk];	// �E����
	    in_s[0][blk + 1] = in[top + blk];		// �E���
	    in_s[blk + 1][0] = in[bot - 1];		// ������
	}
    }
    __syncthreads();

  // ���L�������ɕۑ��������摜�f�[�^���猻�݉�f�ɑ΂���t�B���^�o�͂��v�Z
    xy = (blockIdx.y * blk + threadIdx.y) * stride_o
       +  blockIdx.x * blk + threadIdx.x;
    --x_s;
    out[xy] = op(in_s[y_s-1] + x_s, in_s[y_s] + x_s, in_s[y_s+1] + x_s);
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
    op3x3_kernel<<<blocks, threads>>>((const S*)in[1], (T*)out[1],
				      in.stride(), out.stride(), op);

  // ����
    int	top = 1 + threads.y * blocks.y;
    threads.x = threads.y = out.nrow() - top - 1;
    if (threads.x == 0)
	return;
    blocks.x = out.stride() / threads.x;
    blocks.y = 1;
    op3x3_kernel<<<blocks, threads>>>((const S*)in[top], (T*)out[top],
				      in.stride(), out.stride(), op);

  // �E��
    if (threads.x * blocks.x == out.stride())
	return;
    int	lft = out.stride() - threads.x;
    blocks.x = 1;
    op3x3_kernel<<<blocks, threads>>>((const S*)in[top]  + lft,
				      (	     T*)out[top] + lft,
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
