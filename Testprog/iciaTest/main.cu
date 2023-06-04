/*
 *  $Id$
 */
#include <cstdlib>
#include "TU/Warp.h"
#include "TU/cu/ICIA.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
template <class T> Image<T>
warp(const Image<T>& src, double du, double dv, double theta)
{
    Matrix33d	Htinv;
    Htinv[0][0] = Htinv[1][1] = cos(theta);
    Htinv[1][0] = sin(theta);
    Htinv[0][1] = -Htinv[1][0];
    Htinv[2][0] = -du;
    Htinv[2][1] = -dv;
    Htinv[2][2] = 1.0;

    Image<T>	dst(src.width(), src.height());
    Warp	warp;
    warp.initialize(Htinv,
		    src.width(), src.height(), dst.width(), dst.height());
    warp(src.cbegin(), dst.begin());

    return dst;
}

template <class MAP, class T> void
registerImages(const Image<T>& src, const Image<T>& dst,
	       typename MAP::element_type thresh, bool newton)
{
    using namespace	std;
    using Parameters	= typename cu::ICIA<MAP>::Parameters;

    Parameters	params;
    params.newton	   = newton;
    params.sqcolor_thresh  = thresh*thresh;
    params.niter_max	   = 200;

    const cu::Array2<T>	src_d(src);
    const cu::Array2<T>	dst_d(dst);

  // 位置合わせを実行．
    cu::ICIA<MAP>	registration(params);
    MAP			map;
    map.initialize();
    auto		err = registration(src_d, dst_d, map);
    cerr << "RMS-err = " << sqrt(err) << endl;
    cerr << map;

    registration.print(cerr);
}

}

/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	TU;

    typedef float	T;

    const double	DegToRad = M_PI / 180.0;
    enum Algorithm	{PROJECTIVE, AFFINE, RIGID};
    Algorithm		algorithm = PROJECTIVE;
    double		du = 0.0, dv = 0.0, theta = 0.0;
    T			thresh = 15.0;
    bool		newton = false;
    extern char		*optarg;
    for (int c; (c = getopt(argc, argv, "ARu:v:t:n:T:")) != -1; )
	switch (c)
	{
	  case 'A':
	    algorithm = AFFINE;
	    break;
	  case 'R':
	    algorithm = RIGID;
	    break;
	  case 'u':
	    du = atof(optarg);
	    break;
	  case 'v':
	    dv = atof(optarg);
	    break;
	  case 't':
	    theta = DegToRad * atof(optarg);
	    break;
	  case 'n':
	    newton = true;
	    break;
	  case 'T':
	    thresh = atof(optarg);
	    break;
	}

    try
    {
	std::cerr << "Restoring image...";
	Image<T>	src;
	src.restore(std::cin);
	std::cerr << "done." << std::endl;

	std::cerr << "Warping image...";
	Image<T>	dst = warp(src, du, dv, theta);
	std::cerr << "done." << std::endl;

	src.save(std::cout);
	dst.save(std::cout);

	switch (algorithm)
	{
	  case RIGID:
	    registerImages<cu::Rigidity<T, 2> >(src, dst, thresh, newton);
	    break;
	  case AFFINE:
	    registerImages<cu::Affinity<T, 2, 2> >(src, dst, thresh, newton);
	    break;
	  default:
	    registerImages<cu::Projectivity<T, 2, 2> >(src, dst,
						       thresh, newton);
	    break;
	}
    }
    catch (std::exception& err)
    {
	std::cerr << err.what() << std::endl;
	return 1;
    }

    return 0;
}
