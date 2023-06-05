/*
 *  $Id$
 */
#include <cstdlib>
#include "TU/Warp.h"
#include "TU/cu/ICIA.h"

namespace TU
{
namespace cu
{
Rigidity<float, 2>
createRigidity(float u0, float du, float v0, float dv, float theta)
{
    const auto	c = std::cos(theta);
    const auto	s = std::sin(theta);

    return {{{c, -s}, {s,  c}},
	    {(1 - c)*u0 + s*v0 + du, (1 - c)*v0 - s*u0 + dv}};
}
}	// namespace cu
    
template <class MAP, class T> void
registerImages(const Image<T>& src, T du, T dv, T theta,
	       T thresh, bool newton)
{
    using Parameters	= typename cu::ICIA<MAP>::Parameters;

    const cu::Array2<T>	src_d(src);
    cu::Array2<T>	dst_d(src.nrow(), src.ncol());
    const auto		op = cu::createRigidity(src.ncol()/2, du,
						src.nrow()/2, dv, theta);
    cu::warp(src_d, dst_d.begin(), op);
#if 0
    Image<T>		dst(dst_d);
    src.save(std::cout);
    dst.save(std::cout);			// 結果画像をセーブ
#endif
  // 位置合わせを実行．
    Parameters	params;
    params.newton	   = newton;
    params.sqcolor_thresh  = thresh*thresh;
    params.niter_max	   = 200;

    cu::ICIA<MAP>	registration(params);
    MAP			map;
    map.initialize();
    const auto		err = registration(src_d, dst_d, map);
    std::cerr << "RMS-err = " << std::sqrt(err) << std::endl;
    std::cerr << map;

    registration.print(std::cerr);
}
}	// namespace TU

/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	TU;

    using T = float;

    const float		DegToRad = M_PI / 180.0;
    enum Algorithm	{PROJECTIVE, AFFINE, RIGID};
    Algorithm		algorithm = PROJECTIVE;
    float		du = 0.0, dv = 0.0, theta = 0.0;
    T			thresh = 50.0;
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

	switch (algorithm)
	{
	  case RIGID:
	    registerImages<cu::Rigidity<T, 2> >(src, du, dv, theta,
						thresh, newton);
	    break;
	  case AFFINE:
	    registerImages<cu::Affinity<T, 2, 2> >(src, du, dv, theta,
						   thresh, newton);
	    break;
	  default:
	    registerImages<cu::Projectivity<T, 2, 2> >(src, du, dv, theta,
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
