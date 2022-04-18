#include "TU/cu/Array++.h"
#include "TU/cu/array.h"

namespace TU
{
namespace cu
{
namespace device
{
__global__ void
printTest()
{
    printf("tid=%d\n", threadIdx.x);
}

template <class T, size_t D> __global__ void
arrayTest(array<T, D> a)
{
    const int	i = threadIdx.x;

    printf("tid=%d\n", i);

    if (i < a.size())
	printf("%f\n", a[i]);
}

}	// namespace device

template <class T, size_t D> void
doJob()
{
    std::cerr << "OK" << std::endl;

    Array<device::array<T, D> >	A(1);
    auto&		a = A[0];
    for (size_t i = 0; i < a.size(); ++i)
	a[i] = i;
    device::arrayTest<<<1, D>>>(a);
}

}	// namespace cu
}	// namespace TU

int
main()
{
    TU::cu::doJob<float, 3>();

    return 0;
}
