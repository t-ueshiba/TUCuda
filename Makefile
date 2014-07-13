#
#  $Id: Makefile,v 1.20 2012-09-15 07:21:00 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include
INCDIRS		= -I. -I$(PREFIX)/include -I$(CUDAHOME)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= #-DNDEBUG
CFLAGS		= -g
NVCCFLAGS	= -g
ifeq ($(CXX), icpc)
#  CFLAGS	= -O3
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
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/functional.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/mmInstructions.h \
		/usr/local/include/TU/tuple.h
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
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h \
	TU/CudaUtility.h
CudaGaussianConvolver.o: TU/CudaGaussianConvolver.h TU/CudaFilter.h \
	TU/CudaArray++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h
cudaOp3x3.o: TU/CudaUtility.h TU/CudaArray++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h
cudaSubsample.o: TU/CudaUtility.h TU/CudaArray++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h
cudaSuppressNonExtrema3x3.o: TU/CudaUtility.h TU/CudaArray++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/functional.h \
	/usr/local/include/TU/mmInstructions.h /usr/local/include/TU/tuple.h
