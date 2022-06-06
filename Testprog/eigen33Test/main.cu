/*
 * $Id$
 */
#include "TU/cu/algorithm.h"
#include "TU/cu/Array++.h"
#include "eigen33_gpu.h"
#include "eigen33_cpu.h"

namespace TU
{
template <class T> void
doJob()
{
    std::cerr << ">> ";
    for (cu::mat<T, 3, 3> A; std::cin >> A.x.x; )
    {
	std::cin >> A.x.y >> A.x.z >> A.y.y >> A.y.z >> A.z.z;
	A.y.x = A.x.y;
	A.z.x = A.x.z;
	A.z.y = A.y.z;
	std::cerr << "  A = " << A << std::endl << std::endl;

	cu::mat<T, 3, 3>	Qt;
	cu::vec<T, 3>		d;
	cu::vec<T, 3>		e;
	cu::tridiagonal33(A, Qt, d, e);

	std::cerr << "  d  = " << d << std::endl;
	std::cerr << "  e  = " << e << std::endl;
	std::cerr << "  Qt = " << Qt << std::endl;
	std::cerr << "  Qt*Q = " << dot(Qt, Qt.transpose())
		  << std::endl;
	std::cerr << "  Qt*A*Q = " << dot(dot(Qt, A), Qt.transpose())
		  << std::endl;
	std::cerr << "  det(Qt) = " << dot(Qt.x, cross(Qt.y, Qt.z))
		  << std::endl << std::endl;

	auto	w = cu::qr33(A, Qt);

	std::cerr << "  w = " << w << std::endl;
	std::cerr << "  Qt = " << Qt << std::endl;
	std::cerr << "  Qt*Q = " << dot(Qt, Qt.transpose()) << std::endl;
	std::cerr << "  Qt*A*Q = " << dot(dot(Qt, A), Qt.transpose())
		  << std::endl
		  << "  det(Qt) = " << dot(Qt.x, cross(Qt.y, Qt.z))
		  << std::endl << std::endl;

      //============================================================
	std::cerr << "\n======================" << std::endl;
	T	matA[3][3] = {{A.x.x, A.x.y, A.x.z},
			      {A.y.x, A.y.y, A.y.z},
			      {A.z.x, A.z.y, A.z.z}};
	T	evalues[3];
	::cardano(matA, evalues);
	std::cerr << "  evalues = " << evalues[0]
		  << ' ' << evalues[1] << ' ' << evalues[2] << std::endl;



	T	matQt[3][3], vecd[3], vece[3];
	::tridiagonal33(matA, matQt, vecd, vece);

	Matrix<T>	AA(&matA[0][0], 3, 3), QQt(&matQt[0][0], 3, 3);

	std::cerr << "--- Q ---\n" << QQt;
	std::cerr << "--- Qt*A*Q ---\n" << QQt * AA * transpose(QQt);

	std::cerr << "d = ";
	for (int i = 0; i < 3; ++i)
	    std::cerr << ' ' << vecd[i];
	std::cerr << std::endl;

	std::cerr << "e = ";
	for (int i = 0; i < 3; ++i)
	    std::cerr << ' ' << vece[i];
	std::cerr << std::endl << std::endl;

	::qr33(matA, matQt, evalues);
	std::cerr << "  evalues = " << evalues[0]
		  << ' ' << evalues[1] << ' ' << evalues[2] << std::endl;
	std::cerr << "--- Qt*A*Q ---\n" << QQt * AA * transpose(QQt);

	std::cerr << ">> ";
    }
}

namespace cu
{
namespace device
{
  template <class T> __global__ void
  get_Tgm(const Moment<T>* moment, Rigidity<T, 3>* transform)
  {
      *transform = moment->get_transform();
  }

  template <class T> __global__ void
  get_evalues(const cu::Moment<T>* moment,
	      cu::mat3x<T, 1>* evalues, cu::mat3x<T, 3>* evecs)
  {
      const auto	C = moment->covariance();
    //*evalues = cardano(C);
      eigen33(C, *evecs, *evalues);
  }
}
	
template <class T> void
doJob()
{
    using mat43_t	= mat<T, 4, 3>;
    using rigidity_t	= Rigidity<T, 3>;
    using moment_t	= Moment<T>;
    
    std::cerr << ">> ";
    mat43_t	A;
    std::cin >> A.x.x >> A.x.y >> A.x.z
	     >> A.y.x >> A.y.y >> A.y.z
	     >> A.z.x >> A.z.y >> A.z.z
	     >> A.w.z;
    moment_t		M(A);
    Array<moment_t>	Md(1);
    Md[0] = M;
    
    Array<rigidity_t>	Td(1);
    device::get_Tgm<<<1, 1>>>(Md.data().get(), Td.data().get());
    rigidity_t	Tgm = Td[0];
    std::cerr << "Tgm=" << Tgm << std::endl;

    std::cerr << "cov=" << M.covariance() << std::endl;
    Array<vec<T, 3> >	evalues_d(1);
    Array<mat3x<T, 3> >	evecs_d(1);
    device::get_evalues<<<1, 1>>>(Md.data().get(), evalues_d.data().get(),
				  evecs_d.data().get());
    std::cerr << "evalues=" << evalues_d[0] << std::endl;
    std::cerr << "evecs=" << evecs_d[0] << std::endl;
}

}	// mamespace 
}	// namespace TU

int
main()
{

  //TU::doJob<float>();
    TU::cu::doJob<float>();
    return 0;
}
