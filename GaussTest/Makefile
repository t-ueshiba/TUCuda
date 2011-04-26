#
#  $Id: Makefile,v 1.3 2011-04-26 06:39:30 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I.. -I../FilterTest -I$(PREFIX)/include -I$(CUDAHOME)/include -I$(CUDASDK)/common/inc

PROGRAM		= filterImage
LIBS		= -lTUThread++ -lTUTools++ -lTUCuda++
ifeq ($(OSTYPE), darwin)
  LIBS	       += -lcutil_x86_64
  CCC		= g++
else
  LIBS	       += -lcutil_$(shell uname -m)
endif

CCC		= g++
CPPFLAGS	=
CFLAGS		= -g #-O
NVCCFLAGS	= -g #-O
ifeq ($(CCC), icpc)
  CFLAGS	= -O3
  CPPFLAGS     += -DSSSE3
endif
CCFLAGS		= $(CFLAGS)

LDFLAGS		= -L$(CUDASDK)/lib
LINKER		= $(NVCC)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC
EXTHDRS		= ../FilterTest/filterImageGold.h \
		../TU/CudaFilter.h \
		../TU/CudaGaussianConvolver.h \
		../TU/CudaUtility.h \
		../TU/TU/CudaArray++.h \
		/usr/local/include/TU/Array++.h \
		/usr/local/include/TU/GaussianConvolver.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/IIRFilter.h \
		/usr/local/include/TU/IIRFilterMT.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Normalize.h \
		/usr/local/include/TU/Profiler.h \
		/usr/local/include/TU/Thread++.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/mmInstructions.h \
		/usr/local/include/TU/types.h \
		/usr/local/include/TU/utility.h
HDRS		=
SRCS		= CudaFilter.cu \
		main.cc
OBJS		= CudaFilter.o \
		main.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.3 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk

###
CudaFilter.o: ../TU/CudaFilter.h ../TU/TU/CudaArray++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	../TU/CudaUtility.h
main.o: /usr/local/include/TU/Image++.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/utility.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Normalize.h /usr/local/include/TU/Minimize.h \
	/usr/local/include/TU/Profiler.h /usr/local/include/TU/IIRFilterMT.h \
	/usr/local/include/TU/IIRFilter.h \
	/usr/local/include/TU/mmInstructions.h \
	/usr/local/include/TU/Thread++.h \
	/usr/local/include/TU/GaussianConvolver.h \
	../FilterTest/filterImageGold.h ../TU/CudaGaussianConvolver.h \
	../TU/CudaFilter.h ../TU/TU/CudaArray++.h
