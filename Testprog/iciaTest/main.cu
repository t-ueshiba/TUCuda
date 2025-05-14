/*
 *  $Id$
 */
#include <cstdlib>
#include <fstream>
#include "TU/Warp.h"
#include "TU/cu/ICIA.h"

namespace TU
{
namespace cu
{
template <class T> Rigidity<T, 2>
createRigidity(T u0, T du, T v0, T dv, T theta)
{
    const auto	c = std::cos(theta);
    const auto	s = std::sin(theta);

    return {{{c, -s}, {s,  c}},
	    {(1 - c)*u0 + s*v0 + du, (1 - c)*v0 - s*u0 + dv}};
}
}	// namespace cu

template <class MAP, class C, class T> void
registerImages(const Image<C>& src, T du, T dv, T theta, T thresh)
{
    using Parameters	= typename cu::ICIA<MAP, C>::Parameters;

    cu::Array2<C>	src_d(src);
    cu::Array2<C>	dst_d(src.nrow(), src.ncol());
    const auto		tfm = cu::createRigidity(T(src.ncol()/2), du,
						 T(src.nrow()/2), dv, theta);
    cu::warp(src_d, dst_d.begin(), tfm);
#if 0
    Image<C>		dst(dst_d);
    src.save(std::cout);
    dst.save(std::cout);			// 結果画像をセーブ
    return;
#endif
    
  // 位置合わせを実行．
    Parameters	params;
    params.color_thresh = thresh;

    cu::ICIA<MAP, C>	registration(params);
    MAP			Mds;
    Mds.initialize();
    const auto		mse = registration(src_d, dst_d, Mds).mse;
    std::cerr << "RMS-err = " << std::sqrt(mse) << std::endl;
    std::cerr << Mds;

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

    using C = float;
    using T = float;

    const T		DegToRad = M_PI / 180.0;
    enum Algorithm	{PROJECTIVE, AFFINE, RIGID};
    Algorithm		algorithm = PROJECTIVE;
    T			du = 3.0, dv = -2.0, theta = DegToRad * 3.0,
			thresh = 50.0;
    extern char		*optarg;
    for (int c; (c = getopt(argc, argv, "PARu:v:t:T:")) != -1; )
	switch (c)
	{
	  case 'P':
	    algorithm = PROJECTIVE;
	    break;
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
	  case 'T':
	    thresh = atof(optarg);
	    break;
	}

    try
    {
	std::cerr << "Restoring image...";
	Image<C>	src;
#if 1
	src.restore(std::cin);
#else
	std::ifstream	fin("../ueshiba.epbm");
	src.restore(fin);
#endif
	std::cerr << "done." << std::endl;

	switch (algorithm)
	{
	  case RIGID:
	    registerImages<cu::Rigidity<T, 2> >(src, du, dv, theta, thresh);
	    break;
	  case AFFINE:
	    registerImages<cu::Affinity<T, 2, 2> >(src, du, dv, theta, thresh);
	    break;
	  default:
	    registerImages<cu::Projectivity<T, 2, 2> >(src, du, dv, theta,
						       thresh);
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
