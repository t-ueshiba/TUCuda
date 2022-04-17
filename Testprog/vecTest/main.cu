#include "TU/cu/vec.h"

namespace TU
{
namespace cu
{
template <class T> void
doJob()
{
    mat3x<T, 3>	a{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    mat3x<T, 3>	b{{10, 20, 30}, {40, 50, 60}, {70, 80, 90}};
    vec<T, 3>	v{1, 2, 3};
    
    std::cout << size0<mat2x<T, 3> >() << std::endl;
    
    const auto	c = a + b;

    std::cout << c << std::endl;

    std::cout << c.x + c.y << std::endl;

    std::cout << 3 * v << std::endl;
}
    
}
}


int
main()
{
    TU::cu::doJob<float>();

    return 0;
}


