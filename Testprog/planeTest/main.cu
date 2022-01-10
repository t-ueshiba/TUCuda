/*
 * $Id$
 */
#include <vector>
#include <iostream>
#include <thrust/device_vector.h>
#include "TU/cuda/functional.h"

namespace TU
{
namespace cuda
{
namespace device
{
  template <class T> __global__ void
  fit_plane(const vec<T, 3>* ps, const vec<T, 3>* pe,
	    mat4x<T, 3>* sum, mat3x<T, 3>* plane)
  {
      *sum = 0;
      
      plane_moment_generator<T>	gen;
      for (; ps != pe; ++ps)
	  *sum += gen(*ps);

      plane_estimator<T>	estimator;
      *plane = estimator(*sum);
  }
}
    
template <class T> void
doJob()
{
    for (;;)
    {
	std::vector<vec<T, 3> >	points;
    
	std::cerr << "point>> ";
	for (vec<T, 3> point; std::cin >> point.x; )
	{
	    std::cin >> point.y >> point.z;
	    points.push_back(point);

	    std::cerr << "point>> ";
	}
	if (points.size() < 3)
	    break;

	thrust::device_vector<vec<T, 3> >	points_d(points.size());
	thrust::copy(points.begin(), points.end(), points_d.begin());
	thrust::device_vector<mat4x<T, 3> >	sum_d(1);
	thrust::device_vector<mat3x<T, 3> >	plane_d(1);

	device::fit_plane<<<1, 1>>>(points_d.data().get(),
				    (points_d.data() + points_d.size()).get(),
				    sum_d.data().get(), plane_d.data().get());

	mat4x<T, 3>	sum;
	thrust::copy(sum_d.begin(), sum_d.end(), &sum);
	mat3x<T, 3>	plane;
	thrust::copy(plane_d.begin(), plane_d.end(), &plane);
	
	std::cerr << "---- sum ----\n" << sum << std::endl;
	std::cerr << "---- plane ----\n" << plane << std::endl << std::endl;
    }
}
}	// namespace cuda
}	// namespace TU

int
main()
{
    TU::cuda::doJob<float>();
    return 0;
}
