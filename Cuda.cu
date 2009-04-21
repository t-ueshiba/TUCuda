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
//! CUDAの初期化
/*!
  \param argc	コマンド自身を含んだコマンド行の引数の数
  \param argv	コマンド自身を含んだコマンド行の引数のリスト
 */
void
initializeCUDA(int argc, char* argv[])
{
    CUT_DEVICE_INIT(argc, argv);
}
    
}
