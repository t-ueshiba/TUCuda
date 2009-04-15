/*
 *  $Id: Cuda.cu,v 1.1 2009-04-15 00:32:05 ueshiba Exp $
 */
#include <cstdio>
#include <cutil.h>
#include "TU/Cuda++.h"

namespace TU
{
/************************************************************************
*   Global functions							*
************************************************************************/
void
initializeCUDA(int argc, char* argv[])
{
    CUT_DEVICE_INIT(argc, argv);
}
    
}
