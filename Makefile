#
#  $Id: Makefile,v 1.10 2011-04-18 08:16:55 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(CUDAHOME)/include/TU
INCDIRS		= -I. -I$(PREFIX)/include -I$(CUDAHOME)/include

NAME		= $(shell basename $(PWD))

ifeq ($(OSTYPE), darwin)
    CCC		= g++
endif

CPPFLAGS	= -D_DEBUG #-DNO_BORDER
CFLAGS		= -O
CCFLAGS		= $(CFLAGS)
NVCCFLAGS	= -O

LINKER		= $(NVCC)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/types.h \
		TU/CudaFilter.h \
		TU/CudaGaussianConvolver.h \
		TU/CudaUtility.h \
		TU/TU/CudaArray++.h
HDRS		= CudaArray++.h \
		CudaFilter.h \
		CudaGaussianConvolver.h \
		CudaUtility.h
SRCS		= CudaFilter.cu \
		CudaGaussianConvolver.cc \
		cudaOp3x3.cu \
		cudaSubsample.cu
OBJS		= CudaFilter.o \
		CudaGaussianConvolver.o \
		cudaOp3x3.o \
		cudaSubsample.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.10 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
CudaFilter.o: TU/CudaFilter.h TU/TU/CudaArray++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	TU/CudaUtility.h
CudaGaussianConvolver.o: TU/CudaGaussianConvolver.h TU/CudaFilter.h \
	TU/TU/CudaArray++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h
cudaOp3x3.o: TU/CudaUtility.h TU/TU/CudaArray++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
cudaSubsample.o: TU/CudaUtility.h TU/TU/CudaArray++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
