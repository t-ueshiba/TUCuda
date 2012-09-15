#
#  $Id: Makefile,v 1.19 2012-09-15 05:05:50 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include
INCDIRS		= -I. -I$(PREFIX)/include -I$(CUDAHOME)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	=
CFLAGS		= -g
NVCCFLAGS	= -g
ifeq ($(CXX), icpc)
  CFLAGS	= -O3
  NVCCFLAGS	= -O		# -O2以上にするとコンパイルエラーになる．
  CPPFLAGS     += -DSSE3
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(NVCC)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC .cpp:sC
EXTHDRS		=
HDRS		= TU/CudaArray++.h \
		TU/CudaFilter.h \
		TU/CudaGaussianConvolver.h \
		TU/CudaTexture.h \
		TU/CudaUtility.h
SRCS		= CudaFilter.cu \
		CudaGaussianConvolver.cc \
		cudaOp3x3.cu \
		cudaSubsample.cu \
		cudaSuppressNonExtrema3x3.cu
OBJS		= CudaFilter.o \
		CudaGaussianConvolver.o \
		cudaOp3x3.o \
		cudaSubsample.o \
		cudaSuppressNonExtrema3x3.o

include $(PROJECT)/lib/l.mk
###
CudaFilter.o: TU/CudaFilter.h TU/CudaArray++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	TU/CudaUtility.h
CudaGaussianConvolver.o: TU/CudaGaussianConvolver.h TU/CudaFilter.h \
	TU/CudaArray++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h
cudaOp3x3.o: TU/CudaUtility.h TU/CudaArray++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
cudaSubsample.o: TU/CudaUtility.h TU/CudaArray++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
cudaSuppressNonExtrema3x3.o: TU/CudaUtility.h TU/CudaArray++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
