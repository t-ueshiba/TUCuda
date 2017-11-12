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
  template <class T> __global__ void
  halfTest(const T* in, T* out)
  {
      const auto	x = blockIdx.x*blockDim.x + threadIdx.x;
      
      in  += x;
      out += x;

      auto	val = *in;

      *out = val * val;
  }
}	// namespace device

template <class T> void
halfTest(const Array<T>& in, Array<T>& out)
{
    device::halfTest<T><<<1, in.size()>>>(in.cbegin().get(), out.begin().get());
}
    
}	// namespace cuda

template <class T> void
doJob()
{
    Array<T>		a({1.1f, 2.0f, 3.3f, 4.4f});
  //Array<T>		a({1, 2, 3, 4});
    std::cout << a;
    
    cuda::Array<__half>	in(a), out(in.size());
    cuda::halfTest(in, out);

    Array<T>		b(out);
    std::cout << b;
}
}	// namespace TU

int
main()
{
    TU::doJob<float>();
    return 0;
}
