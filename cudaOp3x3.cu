/*
 *  $Id: cudaOp3x3.cu,v 1.1 2011-04-18 08:16:55 ueshiba Exp $
 */
#include "TU/CudaUtility.h"

namespace TU
{
/************************************************************************
*  global constatnt variables						*
************************************************************************/
static const int	BlockDim = 16;	// u_int�ɂ����CUDA�̃o�O�𓥂ށI
    
/************************************************************************
*  device functions							*
************************************************************************/
template <class T, class OP> static __global__ void
op3x3_kernel(const T* in, float* out, u_int stride_i, u_int stride_o, OP op3x3)
{
  // ���̃J�[�l���̓u���b�N���E�����̂��߂� blockDim.x == blockDim.y ������
    int	lft = (blockIdx.y * blockDim.y + threadIdx.y) * stride_i 
	    +  blockIdx.x * blockDim.x,			// ���݈ʒu���猩�č��[
	xy  = lft + threadIdx.x;			// ���݈ʒu
    int	x_s = threadIdx.x + 1,
	y_s = threadIdx.y + 1;

  // ���摜�̃u���b�N��������т��̊O�g1��f�������L�������ɓ]��
    __shared__ float	in_s[BlockDim+2][BlockDim+2];

    in_s[y_s][x_s] = in[xy];				// ����

    if (threadIdx.y == 0)	// �u���b�N�̏�[?
    {

	int	top = xy - stride_i;			// ���݈ʒu�̒���
	int	bot = xy + blockDim.y * stride_i;	// ���݈ʒu���猩�ĉ��[
	in_s[	      0][x_s] = in[top];		// ��g
	in_s[BlockDim+1][x_s] = in[bot];		// ���g

	lft += threadIdx.x * stride_i;
	in_s[x_s][	   0] = in[lft - 1];		// ���g
	in_s[x_s][BlockDim+1] = in[lft + BlockDim];	// �E�g

	if (threadIdx.x == 0)	// �u���b�N�̍����?
	{
	    if ((blockIdx.x != 0) || (blockIdx.y != 0))
	  	in_s[0][0] = in[top - 1];			    // �����
	    if ((blockIdx.x != gridDim.x - 1) || (blockIdx.y != gridDim.y - 1))
		in_s[BlockDim+1][BlockDim+1] = in[bot + BlockDim];  // �E����
	    in_s[0][BlockDim+1] = in[top + BlockDim];		    // �E���
	    in_s[BlockDim+1][0] = in[bot - 1];			    // ������
	}
    }
    __syncthreads();

  // ���L�������ɕۑ��������摜�f�[�^���猻�݉�f�ɑ΂���t�B���^�o�͂��v�Z
    int	xy_o = (blockIdx.y * blockDim.y + threadIdx.y) * stride_o
	     +  blockIdx.x * blockDim.x + threadIdx.x;
    --x_s;
    out[xy_o] = op3x3(in_s[y_s-1] + x_s, in_s[y_s] + x_s, in_s[y_s+1] + x_s);
}

/************************************************************************
*  global functions							*
************************************************************************/
template <class T, class OP> inline static void
cudaOp3x3(const CudaArray2<T>& in, CudaArray2<float>& out, OP op)
{
    using namespace	std;
    
    if (in.nrow() < 3 || in.ncol() < 3)
	return;
    
    out.resize(in.nrow(), in.ncol());

  // �ŏ��ƍŌ�̍s�������� (in.nrow() - 2) x in.ncol() �̔z��Ƃ��Ĉ���
    dim3	threads(BlockDim, BlockDim);
    dim3	blocks(in.ncol()/threads.x, (in.nrow() - 2)/threads.y);
    
  // ����
    op3x3_kernel<<<blocks, threads>>>((const T*)in[1], (float*)out[1],
				      in.stride(), out.stride(), op);
#ifndef NO_BORDER
  /*  // �E��
    uint	offset = threads.y * blocks.y;
    threads.x = threads.y = in.ncol() - offset;
    blocks.x = 1;
    blocks.y = (in.nrow() - 2)/threads.y;
    op3x3_kernel<<<blocks, threads>>>((const T*)in[1]  + offset,
				      (  float*)out[1] + offset,
				      in.stride(), out.stride(), op);


    dim3	threads(BlockDim, BlockDim);
    for (int i = 2; i < in.nrow(); )
    {
	threads.x = threads.y = std::min(in.nrow() - i, in.ncol() - j);
	blocks.y = (in.nrow() - i) / threads.y;
	


	i += threads.y * blocks.y;
	}*/
#endif
}

template void
cudaOp3x3(const CudaArray2<u_char>& in,
		CudaArray2<float>& out, laplacian3x3<float> op)		;
template void
cudaOp3x3(const CudaArray2<float>& in,
		CudaArray2<float>& out, laplacian3x3<float> op)		;
template void
cudaOp3x3(const CudaArray2<u_char>& in,
		CudaArray2<float>& out, det3x3<float> op)		;
template void
cudaOp3x3(const CudaArray2<float>& in,
		CudaArray2<float>& out, det3x3<float> op)		;

}
