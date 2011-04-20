/*
 * $Id: CudaFilter.cu,v 1.5 2011-04-20 08:15:07 ueshiba Exp $
 */
#include "TU/CudaFilter.h"
#include "TU/CudaUtility.h"

namespace TU
{
/************************************************************************
*  global constatnt variables						*
************************************************************************/
static const u_int		BlockDimX = 32;
static const u_int		BlockDimY = 16;
    
static __constant__ float	_lobeH[CudaFilter2::LOBE_SIZE_MAX];
static __constant__ float	_lobeV[CudaFilter2::LOBE_SIZE_MAX];

/************************************************************************
*  device functions							*
************************************************************************/
template <uint D, class S, class T> static __global__ void
filter_kernel(const S* in, T* out, uint stride_i, uint stride_o)
{
    extern __shared__ float	in_s[];
    int				xy, dxy, xy_s, dxy_s, xy_o;
    const float*		lobe;
    
  // D = 2*����Ĺ (�������ˤޤ��� D = 2*����Ĺ + 1�ʽ�������
    if (D & 0x1)	// �������˥ե��륿���
    {
	int	x = (blockIdx.x + 1)*blockDim.x + threadIdx.x,
		y = (blockIdx.y + 1)*blockDim.y + threadIdx.y;

	xy    = y*stride_i + x;
	dxy   = blockDim.y*stride_i;
	
	xy_o  = y*stride_o + x;
	
      // bank conflict���ɤ����ᡤin_s[]���:blockDim.x, ��:3*blockDim.y + 1 ��
      // 2��������Ȥ��ư�����
	xy_s  = threadIdx.x*(3*blockDim.y + 1) + blockDim.y + threadIdx.y;
	dxy_s = blockDim.y;

	lobe = _lobeV;
    }
    else		// �������˥ե��륿���
    {
	int	x = (blockIdx.x + 1)*blockDim.x + threadIdx.x,
		y = blockIdx.y	    *blockDim.y + threadIdx.y;

	xy    = y*stride_i + x;
	dxy   = blockDim.x;

	xy_o  = y*stride_o + x;

      // in_s[]���:blockDim.y, ��:3*blockDim.x ��2��������Ȥ��ư�����
    	xy_s  = threadIdx.y*(3*blockDim.x) + blockDim.x + threadIdx.x;
	dxy_s = blockDim.x;

	lobe = _lobeH;
    }
    
  // ��������3�ĤΥ�����(����åɥ֥�å����б�)��ͭ����˥��ԡ�
    in_s[xy_s - dxy_s] = in[xy - dxy];
    in_s[xy_s	     ] = in[xy	    ];
    in_s[xy_s + dxy_s] = in[xy + dxy];
    __syncthreads();
    
  // ���±黻
    switch (D >> 1)
    {
      case 17:	// ����Ĺ��17���Ǥζ��ؿ����߹��ߥ����ͥ�
	out[xy_o] = lobe[ 0] * (in_s[xy_s - 16] + in_s[xy_s + 16])
		  + lobe[ 1] * (in_s[xy_s - 15] + in_s[xy_s + 15])
		  + lobe[ 2] * (in_s[xy_s - 14] + in_s[xy_s + 14])
		  + lobe[ 3] * (in_s[xy_s - 13] + in_s[xy_s + 13])
		  + lobe[ 4] * (in_s[xy_s - 12] + in_s[xy_s + 12])
		  + lobe[ 5] * (in_s[xy_s - 11] + in_s[xy_s + 11])
		  + lobe[ 6] * (in_s[xy_s - 10] + in_s[xy_s + 10])
		  + lobe[ 7] * (in_s[xy_s -  9] + in_s[xy_s +  9])
		  + lobe[ 8] * (in_s[xy_s -  8] + in_s[xy_s +  8])
		  + lobe[ 9] * (in_s[xy_s -  7] + in_s[xy_s +  7])
		  + lobe[10] * (in_s[xy_s -  6] + in_s[xy_s +  6])
		  + lobe[11] * (in_s[xy_s -  5] + in_s[xy_s +  5])
		  + lobe[12] * (in_s[xy_s -  4] + in_s[xy_s +  4])
		  + lobe[13] * (in_s[xy_s -  3] + in_s[xy_s +  3])
		  + lobe[14] * (in_s[xy_s -  2] + in_s[xy_s +  2])
		  + lobe[15] * (in_s[xy_s -  1] + in_s[xy_s +  1])
		  + lobe[16] *  in_s[xy_s     ];
	break;
      case 16:	// ����Ĺ��16���Ǥδ�ؿ����߹��ߥ����ͥ�
	out[xy_o] = lobe[ 0] * (in_s[xy_s - 16] - in_s[xy_s + 16])
		  + lobe[ 1] * (in_s[xy_s - 15] - in_s[xy_s + 15])
		  + lobe[ 2] * (in_s[xy_s - 14] - in_s[xy_s + 14])
		  + lobe[ 3] * (in_s[xy_s - 13] - in_s[xy_s + 13])
		  + lobe[ 4] * (in_s[xy_s - 12] - in_s[xy_s + 12])
		  + lobe[ 5] * (in_s[xy_s - 11] - in_s[xy_s + 11])
		  + lobe[ 6] * (in_s[xy_s - 10] - in_s[xy_s + 10])
		  + lobe[ 7] * (in_s[xy_s -  9] - in_s[xy_s +  9])
		  + lobe[ 8] * (in_s[xy_s -  8] - in_s[xy_s +  8])
		  + lobe[ 9] * (in_s[xy_s -  7] - in_s[xy_s +  7])
		  + lobe[10] * (in_s[xy_s -  6] - in_s[xy_s +  6])
		  + lobe[11] * (in_s[xy_s -  5] - in_s[xy_s +  5])
		  + lobe[12] * (in_s[xy_s -  4] - in_s[xy_s +  4])
		  + lobe[13] * (in_s[xy_s -  3] - in_s[xy_s +  3])
		  + lobe[14] * (in_s[xy_s -  2] - in_s[xy_s +  2])
		  + lobe[15] * (in_s[xy_s -  1] - in_s[xy_s +  1]);
	break;
      case  9:	// ����Ĺ��9���Ǥζ��ؿ����߹��ߥ����ͥ�
	out[xy_o] = lobe[ 0] * (in_s[xy_s -  8] + in_s[xy_s +  8])
		  + lobe[ 1] * (in_s[xy_s -  7] + in_s[xy_s +  7])
		  + lobe[ 2] * (in_s[xy_s -  6] + in_s[xy_s +  6])
		  + lobe[ 3] * (in_s[xy_s -  5] + in_s[xy_s +  5])
		  + lobe[ 4] * (in_s[xy_s -  4] + in_s[xy_s +  4])
		  + lobe[ 5] * (in_s[xy_s -  3] + in_s[xy_s +  3])
		  + lobe[ 6] * (in_s[xy_s -  2] + in_s[xy_s +  2])
		  + lobe[ 7] * (in_s[xy_s -  1] + in_s[xy_s +  1])
		  + lobe[ 8] *  in_s[xy_s     ];
	break;
      case  8:	// ����Ĺ��8���Ǥδ�ؿ����߹��ߥ����ͥ�
	out[xy_o] = lobe[ 0] * (in_s[xy_s -  8] - in_s[xy_s +  8])
		  + lobe[ 1] * (in_s[xy_s -  7] - in_s[xy_s +  7])
		  + lobe[ 2] * (in_s[xy_s -  6] - in_s[xy_s +  6])
		  + lobe[ 3] * (in_s[xy_s -  5] - in_s[xy_s +  5])
		  + lobe[ 4] * (in_s[xy_s -  4] - in_s[xy_s +  4])
		  + lobe[ 5] * (in_s[xy_s -  3] - in_s[xy_s +  3])
		  + lobe[ 6] * (in_s[xy_s -  2] - in_s[xy_s +  2])
		  + lobe[ 7] * (in_s[xy_s -  1] - in_s[xy_s +  1]);
	break;
      case  5:	// ����Ĺ��5���Ǥζ��ؿ����߹��ߥ����ͥ�
	out[xy_o] = lobe[0] * (in_s[xy_s - 4] + in_s[xy_s + 4])
		  + lobe[1] * (in_s[xy_s - 3] + in_s[xy_s + 3])
		  + lobe[2] * (in_s[xy_s - 2] + in_s[xy_s + 2])
		  + lobe[3] * (in_s[xy_s - 1] + in_s[xy_s + 1])
		  + lobe[4] *  in_s[xy_s    ];
	break;
      case  4:	// ����Ĺ��4���Ǥδ�ؿ����߹��ߥ����ͥ�
	out[xy_o] = lobe[0] * (in_s[xy_s - 4] - in_s[xy_s + 4])
		  + lobe[1] * (in_s[xy_s - 3] - in_s[xy_s + 3])
		  + lobe[2] * (in_s[xy_s - 2] - in_s[xy_s + 2])
		  + lobe[3] * (in_s[xy_s - 1] - in_s[xy_s + 1]);
	break;
      case  3:	// ����Ĺ��2���Ǥζ��ؿ����߹��ߥ����ͥ�
	out[xy_o] = lobe[0] * (in_s[xy_s - 2] + in_s[xy_s + 2])
		  + lobe[1] * (in_s[xy_s - 1] + in_s[xy_s + 1])
		  + lobe[2] *  in_s[xy_s    ];
	break;
      case  2:	// ����Ĺ��2���Ǥδ�ؿ����߹��ߥ����ͥ�
	out[xy_o] = lobe[0] * (in_s[xy_s - 2] - in_s[xy_s + 2])
		  + lobe[1] * (in_s[xy_s - 1] - in_s[xy_s + 1]);
	break;
    }
}

/************************************************************************
*  static functions							*
************************************************************************/
template <uint D, class S, class T> inline static void
convolve_dispatch(const CudaArray2<S>& in, CudaArray2<T>& out)
{
    dim3	threads(BlockDimX, BlockDimY);
    dim3	blocks(in.ncol()/threads.x - 2,
		       in.nrow()/threads.y - 2*(D & 0x1));
    uint	shmsize = threads.x*(3*threads.y + (D & 0x1))*sizeof(float);

  // �濴��
    filter_kernel<D><<<blocks, threads, shmsize>>>((const S*)in, (T*)out,
						   in.stride(),
						   out.stride());
#ifndef NO_BORDER
  // ��ü�ȱ�ü
    const uint	lobeSize = (D >> 1) & ~0x1;	// �濴����ޤޤʤ�����Ĺ
    uint	offset = (1 + blocks.x)*threads.x - lobeSize;
    blocks.x  = threads.x/lobeSize - 1;
    threads.x = lobeSize;
    filter_kernel<D><<<blocks, threads, shmsize>>>((const S*)in, (T*)out,
						   in.stride(),
						   out.stride());
    filter_kernel<D><<<blocks, threads, shmsize>>>((const S*)in  + offset,
						   (	  T*)out + offset,
						   in.stride(),
						   out.stride());

    if (D & 0x1)	// �������˥ե��륿���
    {
      // ��ü�Ȳ�ü
	offset	  = (1 + blocks.y)*threads.y - lobeSize;
	blocks.x  = in.ncol()/threads.x - 2;
	blocks.y  = threads.y/lobeSize  - 1;
	threads.y = lobeSize;
	filter_kernel<D><<<blocks, threads, shmsize>>>((const S*)in, (T*)out,
						       in.stride(),
						       out.stride());
	filter_kernel<D><<<blocks, threads, shmsize>>>(
	    (const S*)in  + offset*in.stride(),
	    (	   T*)out + offset*out.stride(),
	    in.stride(), out.stride());
    }
#endif
}
    
/************************************************************************
*  class CudaFilter2							*
************************************************************************/
//! CUDA�ˤ��2�����ե��륿���������롥
CudaFilter2::CudaFilter2()
    :_lobeSizeH(0), _lobeSizeV(0)
{
    int	device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&_prop, device);
}

//! 2�����ե��륿�Υ��֤����ꤹ�롥
/*!
  Ϳ������֤�Ĺ���ϡ����߹��ߥ����ͥ뤬���ؿ��ξ��2^n + 1, ��ؿ��ξ��2^n
  (n = 1, 2, 3, 4)�Ǥʤ���Фʤ�ʤ���
  \param lobeH	����������
  \param lobeV	����������
  \return	����2�����ե��륿
*/
CudaFilter2&
CudaFilter2::initialize(const Array<float>& lobeH, const Array<float>& lobeV)
{
    using namespace	std;
    
    if (_lobeSizeH > LOBE_SIZE_MAX || _lobeSizeV > LOBE_SIZE_MAX)
	throw runtime_error("CudaFilter2::initialize: too large lobe size!");
    
    _lobeSizeH = lobeH.size();
    _lobeSizeV = lobeV.size();
    cudaCopyToConstantMemory(lobeH.begin(), lobeH.end(), _lobeH);
    cudaCopyToConstantMemory(lobeV.begin(), lobeV.end(), _lobeV);

    return *this;
}
    
//! Ϳ����줿2��������Ȥ��Υե��륿����߹���
/*!
  \param in	����2��������
  \param out	����2��������
  \return	���Υե��륿����
*/
template <class S, class T> const CudaFilter2&
CudaFilter2::convolve(const CudaArray2<S>& in, CudaArray2<T>& out) const
{
    convolveH(in, _buf);
    convolveV(_buf, out);

    return *this;
}
    
//! Ϳ����줿2��������Ȥ��Υե��륿�������˾��߹���
/*!
  \param in	����2��������
  \param out	����2��������
  \return	���Υե��륿����
*/
template <class S, class T> const CudaFilter2&
CudaFilter2::convolveH(const CudaArray2<S>& in, CudaArray2<T>& out) const
{
    using namespace	std;
    
    out.resize(in.nrow(), in.ncol());

    switch (_lobeSizeH)
    {
      case 17:
	convolve_dispatch<34>(in, out);
	break;
      case 16:
	convolve_dispatch<32>(in, out);
	break;
      case  9:
	convolve_dispatch<18>(in, out);
	break;
      case  8:
	convolve_dispatch<16>(in, out);
	break;
      case  5:
	convolve_dispatch<10>(in, out);
	break;
      case  4:
	convolve_dispatch< 8>(in, out);
	break;
      case  3:
	convolve_dispatch< 6>(in, out);
	break;
      case  2:
	convolve_dispatch< 4>(in, out);
	break;
      default:
	throw runtime_error("CudaFilter2::convolveH: unsupported horizontal lobe size!");
    }

    return *this;
}

//! Ϳ����줿2��������Ȥ��Υե��륿��������˾��߹���
/*!
  \param in	����2��������
  \param out	����2��������
  \return	���Υե��륿����
*/
template <class S, class T> const CudaFilter2&
CudaFilter2::convolveV(const CudaArray2<S>& in, CudaArray2<T>& out) const
{
    using namespace	std;
    
    out.resize(in.nrow(), in.ncol());
    
    switch (_lobeSizeV)
    {
      case 17:
	convolve_dispatch<35>(in, out);
	break;
      case 16:
	convolve_dispatch<33>(in, out);
	break;
      case  9:
	convolve_dispatch<19>(in, out);
	break;
      case  8:
	convolve_dispatch<17>(in, out);
	break;
      case  5:
	convolve_dispatch<11>(in, out);
	break;
      case  4:
	convolve_dispatch< 9>(in, out);
	break;
      case  3:
	convolve_dispatch< 7>(in, out);
	break;
      case  2:
	convolve_dispatch< 5>(in, out);
	break;
      default:
	throw runtime_error("CudaFilter2::convolveV: unsupported vertical lobe size!");
    }

    return *this;
}

template const CudaFilter2&
CudaFilter2::convolve(const CudaArray2<u_char>& in,
			    CudaArray2<u_char>& out)		const	;
template const CudaFilter2&
CudaFilter2::convolve(const CudaArray2<u_char>& in,
			    CudaArray2<float>&  out)		const	;
template const CudaFilter2&
CudaFilter2::convolve(const CudaArray2<float>& in,
			    CudaArray2<u_char>& out)		const	;
template const CudaFilter2&
CudaFilter2::convolve(const CudaArray2<float>& in,
			    CudaArray2<float>& out)		const	;
}
