/*
 *  $Id: main.cc,v 1.2 2009-05-10 23:36:49 ueshiba Exp $
 */
#include <fstream>
#include <stdexcept>
#include "TU/Cuda++.h"
#include "TU/Image++.h"

namespace TU
{
void	interpolate(const Array2<ImageLine<RGBA> >& image0,
		    const Array2<ImageLine<RGBA> >& image1,
			  Array2<ImageLine<RGBA> >& image2);
}

/************************************************************************
*  Global fucntions							*
************************************************************************/
int
main(int argc, char *argv[])
{
    using namespace	std;
    using namespace	TU;
    
    initializeCUDA(argc, argv);

    try
    {
	Image<RGBA>	images[3];

      // Restore a pair of input images.
	fstream		in;
	in.open("src0.ppm");
	if (!in)
	    throw runtime_error("Failed to open src1.ppm!!");
	images[0].restore(in);
	in.close();
	in.open("src1.ppm");
	if (!in)
	    throw runtime_error("Failed to open src2.ppm!!");
	images[1].restore(in);
	in.close();

      // Do main job.
	interpolate(images[0], images[1], images[2]);

      // Save the obtained results.
	images[2].save(cout);
    }
    catch (exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }

    return 0;
}
