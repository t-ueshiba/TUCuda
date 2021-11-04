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
    for (cuda::mat3x<T, 3> A; std::cin >> A.x.x; )
    {
	std::cin >> A.x.y >> A.x.z >> A.y.y >> A.y.z >> A.z.z;
	A.y.x = A.x.y;
	A.z.x = A.x.z;
	A.z.y = A.y.z;
	std::cerr << "  A = " << A << std::endl;

	cuda::mat3x<T, 3>	Qt;
	auto	w = cuda::eigen33(A, Qt);

	std::cerr << "  w = " << w << std::endl;
	std::cerr << "  Qt = " << Qt << std::endl;
	std::cerr << "  Qt*Q = " << dot(Qt, Qt.transpose()) << std::endl;
	std::cerr << "  Qt*A*Q = " << dot(dot(Qt, A), Qt.transpose())
		  << std::endl << std::endl;
									
	cuda::vec<T, 3>		d;
	cuda::vec<T, 2>		e;
	cuda::tridiagonal33(A, Qt, d, e);

	std::cerr << "  d  = " << d << std::endl;
	std::cerr << "  e  = " << e << std::endl;
	std::cerr << "  Qt = " << Qt << std::endl;
	std::cerr << "  Qt*Q = " << dot(Qt, Qt.transpose())
		  << std::endl;
	std::cerr << "  Qt*A*Q = " << dot(dot(Qt, A), Qt.transpose())
		  << std::endl;
								       
      //============================================================
	std::cerr << "\n======================" << std::endl;
	T	matA[3][3] = {{A.x.x, A.x.y, A.x.z},
			      {A.y.x, A.y.y, A.y.z},
			      {A.z.x, A.z.y, A.z.z}};
	T	evalues[3];
	::cardano(matA, evalues);
	std::cerr << "  evalues = " << evalues[0]
		  << ' ' << evalues[1] << ' ' << evalues[2] << std::endl;

							       

	T	matQ[3][3], vecd[3], vece[2];
	::tridiagonal33(matA, matQ, vecd, vece);

	Matrix<T>	Am(&matA[0][0], 3, 3), Qm(&matQ[0][0], 3, 3);
							       
	std::cerr << "--- Q ---\n" << Qm;
	std::cerr << "--- Qt*A*Q ---\n" << Qm * Am * transpose(Qm);

	std::cerr << "d = ";
	for (int i = 0; i < 3; ++i)
	    std::cerr << ' ' << vecd[i];
	std::cerr << std::endl;
					     
	std::cerr << "e = ";
	for (int i = 0; i < 2; ++i)
	    std::cerr << ' ' << vece[i];
	std::cerr << std::endl;
					     
	::eigen33(matA, matQ, evalues);
	std::cerr << "  evalues = " << evalues[0]
		  << ' ' << evalues[1] << ' ' << evalues[2] << std::endl;
	std::cerr << "--- Qt*A*Q ---\n" << Qm * Am * transpose(Qm);
		     
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
