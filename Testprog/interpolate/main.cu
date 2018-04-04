/*
 *  $Id: main.cc,v 1.2 2012-08-30 12:19:21 ueshiba Exp $
 */
#include <fstream>
#include <stdexcept>
#include "TU/Image++.h"
#include "TU/cuda/Array++.h"

namespace TU
{
namespace cuda
{
template <class T> void
interpolate(const Array2<T>& d_image0,
	    const Array2<T>& d_image1, Array2<T>& d_image2);
}
}

/************************************************************************
*  Global fucntions							*
************************************************************************/
int
main(int argc, char *argv[])
{
    using namespace	std;
    using namespace	TU;
    
    try
    {
	typedef u_char	pixel_type;
	
	Image<pixel_type>	image0, image1;

      // Restore a pair of input images.
	fstream		in;
	in.open("src0.ppm");
	if (!in)
	    throw runtime_error("Failed to open src1.ppm!!");
	image0.restore(in);
	in.close();
	in.open("src1.ppm");
	if (!in)
	    throw runtime_error("Failed to open src2.ppm!!");
	image1.restore(in);
	in.close();

      // Do main job.
	cuda::Array2<pixel_type>	d_image0(image0),
					d_image1(image1), d_image2;
	cuda::interpolate(d_image0, d_image1, d_image2);

      // Save the obtained results.
	Image<pixel_type>	image2(d_image2);
	image2.save(cout);
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
