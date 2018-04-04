/*
 * $Id$
 */
#include "TU/cuda/Array++.h"
#include "TU/cuda/fp16.h"

namespace TU
{
namespace cuda
{
namespace device
{
  template <class COL, class COL_O> __global__ void
  halfTest(COL in, COL_O out, int strideI, int strideO)
  {
      const auto	row = blockIdx.y*blockDim.y + threadIdx.y;
      const auto	col = blockIdx.x*blockDim.x + threadIdx.x;
      
      in  += row*strideI + col;
      out += row*strideO + col;

      auto	val = *in;

      *out = val * val;
  }
}	// namespace device

template <class T> void
halfTest(const Array2<T>& in, Array2<T>& out)
{
    dim3	threads(in.ncol(), in.nrow());
    dim3	blocks(1, 1);
    device::halfTest<<<blocks, threads>>>(in.cbegin()->cbegin().get(),
					  out.begin()->begin().get(),
					  in.stride(), out.stride());
}
    
}	// namespace cuda

template <class T> void
doJob()
{
    Array2<T>			a({{1.1f, 2.0f, 3.3f, 4.4f},
				   {5.5f, 6.6f, 7.7f, 8.8f}});
  //Array<T>		a({1, 2, 3, 4});
    std::cout << a;
    
    cuda::Array2<__half>	in(a), out(in.nrow(), in.ncol());
    cuda::halfTest(in, out);

    Array2<T>			b(out);
    std::cout << b;
}
}	// namespace TU

int
main()
{
    TU::doJob<float>();
    return 0;
}
