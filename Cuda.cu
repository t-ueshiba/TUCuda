/*
 *  $Id: Cuda.cu,v 1.2 2009-04-21 23:30:35 ueshiba Exp $
 */
#include <cstdio>
#include <cutil.h>
#include "TU/Cuda++.h"

namespace TU
{
/************************************************************************
*   Global functions							*
************************************************************************/
//! CUDA�̏�����
/*!
  \param argc	�R�}���h���g���܂񂾃R�}���h�s�̈����̐�
  \param argv	�R�}���h���g���܂񂾃R�}���h�s�̈����̃��X�g
 */
void
initializeCUDA(int argc, char* argv[])
{
    CUT_DEVICE_INIT(argc, argv);
}
    
}
