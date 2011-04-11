/*
 *  $Id: main.cc,v 1.3 2011-04-11 08:06:15 ueshiba Exp $
 */
#include <fstream>
#include <stdexcept>
#include "TU/Image++.h"

namespace TU
{
void	interpolate(const Image<RGBA>& image0,
		    const Image<RGBA>& image1,
			  Image<RGBA>& image2);
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
