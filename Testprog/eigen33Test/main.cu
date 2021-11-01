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
	set_zero(Qt);
	std::cerr << Qt << std::endl;
	cuda::vec<T, 3>		w;
	cuda::device::eigen33(A, Qt, w);

	auto	B = A;
	B.x.x -= w.x;
	B.y.y -= w.x;
	B.z.z -= w.x;
	std::cerr << "  det.x = " << cuda::dot(cuda::cross(B.x, B.y), B.z)
		  << std::endl;

	B = A;
	B.x.x -= w.y;
	B.y.y -= w.y;
	B.z.z -= w.y;
	std::cerr << "  det.y = " << cuda::dot(cuda::cross(B.x, B.y), B.z)
		  << std::endl;

	B = A;
	B.x.x -= w.z;
	B.y.y -= w.z;
	B.z.z -= w.z;
	std::cerr << "  det.z = " << cuda::dot(cuda::cross(B.x, B.y), B.z)
		  << std::endl;

	std::cerr << "  w = " << w << std::endl;
	std::cerr << "  Qt = " << Qt << std::endl;
	const auto	Q = cuda::transpose(Qt);
	std::cerr << "  Q  = " << Q << std::endl;
	std::cerr << "  Qt*Q = " << dot(Qt, Q)
		  << std::endl << std::endl;
	std::cerr << "  Qt*A*Q = " << dot(dot(Qt, A), cuda::transpose(Qt))
		  << std::endl;
									
	    
	cuda::vec<T, 3>		d;
	cuda::vec<T, 2>		e;
	cuda::tridiagonal33(A, Qt, d, e);

	std::cerr << "  d  = " << d << std::endl;
	std::cerr << "  e  = " << e << std::endl;
	std::cerr << "  Qt = " << Qt << std::endl;
	std::cerr << "  Qt*Q = " << dot(Qt, cuda::transpose(Qt))
		  << std::endl;
	std::cerr << "  Qt*A*Q = " << dot(dot(Qt, A), cuda::transpose(Qt))
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
	std::cerr << "--- Qt*A*Q ---\n" << transpose(Qm) * Am * Qm;

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
	std::cerr << "--- Qt*A*Q ---\n" << transpose(Qm) * Am * Qm;
		     
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
