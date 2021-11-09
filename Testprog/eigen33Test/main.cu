/*
 * $Id$
 */
#include "TU/cuda/eigen33.h"
#include "eigen33_cpu.h"

namespace TU
{
template <class T> void
doJob()
{
    std::cerr << ">> ";
    for (cuda::mat<T, 3, 3> A; std::cin >> A.x.x; )
    {
	std::cin >> A.x.y >> A.x.z >> A.y.y >> A.y.z >> A.z.z;
	A.y.x = A.x.y;
	A.z.x = A.x.z;
	A.z.y = A.y.z;
	std::cerr << "  A = " << A << std::endl << std::endl;

	cuda::mat<T, 3, 3>	Qt;
	cuda::vec<T, 3>		d;
	cuda::vec<T, 3>		e;
	cuda::tridiagonal33(A, Qt, d, e);

	std::cerr << "  d  = " << d << std::endl;
	std::cerr << "  e  = " << e << std::endl;
	std::cerr << "  Qt = " << Qt << std::endl;
	std::cerr << "  Qt*Q = " << dot(Qt, Qt.transpose())
		  << std::endl;
	std::cerr << "  Qt*A*Q = " << dot(dot(Qt, A), Qt.transpose())
		  << std::endl << std::endl;
								       
	auto	w = cuda::eigen33(A, Qt);

	std::cerr << "  w = " << w << std::endl;
	std::cerr << "  Qt = " << Qt << std::endl;
	std::cerr << "  Qt*Q = " << dot(Qt, Qt.transpose()) << std::endl;
	std::cerr << "  Qt*A*Q = " << dot(dot(Qt, A), Qt.transpose())
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
}	// namespace TU

int
main()
{

    TU::doJob<float>();
    return 0;
}
