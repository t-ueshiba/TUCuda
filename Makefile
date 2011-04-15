#
#  $Id: Makefile,v 1.8 2011-04-15 05:18:52 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/lib
INCDIR		= $(CUDAHOME)/include/TU
INCDIRS		= -I. -I$(PREFIX)/include -I$(CUDAHOME)/include

NAME		= $(shell basename $(PWD))

CPPFLAGS	= -D_DEBUG #-DNO_BORDER
CFLAGS		= -g -O
NVCCFLAGS	= -g
ifeq ($(CCC), icpc)
  CFLAGS	= -O3
  NVCCFLAGS	= -O	# -O2以上にするとコンパイルエラーになる．
  ifeq ($(OSTYPE), darwin)
    CPPFLAGS   += -DSSE3
    CFLAGS     += -xSSE3
  else
    CPPFLAGS   += -DSSE3 
    CFLAGS     += -xSSE3
  endif
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/types.h \
		TU/CudaFilter.h \
		TU/CudaGaussianConvolver.h \
		TU/TU/CudaArray++.h
HDRS		= CudaArray++.h \
		CudaFilter.h \
		CudaGaussianConvolver.h
SRCS		= CudaFilter.cu \
		CudaGaussianConvolver.cc \
		CudaGaussianConvolver.cu \
		cudaSubsample.cu
OBJS		= CudaFilter.o \
		CudaGaussianConvolver.o \
		CudaGaussianConvolver.o \
		cudaSubsample.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.8 $	|		\
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
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h
CudaGaussianConvolver.o: TU/CudaGaussianConvolver.h TU/CudaFilter.h \
	TU/TU/CudaArray++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h
CudaGaussianConvolver.o: TU/CudaGaussianConvolver.h TU/CudaFilter.h \
	TU/TU/CudaArray++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h
cudaSubsample.o: TU/TU/CudaArray++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/types.h
