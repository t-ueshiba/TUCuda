#
#  $Id: Makefile,v 1.4 2012-01-23 06:13:43 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(PREFIX)/bin
INCDIRS		= -I.. -I../FilterTest -I$(PREFIX)/include -I$(CUDAHOME)/include -I$(CUDASDK)/common/inc

PROGRAM		= filterImage
LIBS		= -lTUTools++ -lTUCuda++
ifeq ($(OSTYPE), darwin)
  LIBS	       += -lcutil_x86_64
else
  LIBS	       += -lcutil_$(shell uname -m)
endif

CPPFLAGS	=
CFLAGS		= -O
NVCCFLAGS	= -O
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
EXTHDRS		= ../TU/CudaUtility.h \
		../TU/TU/CudaArray++.h \
		/usr/local/include/TU/Array++.h \
		/usr/local/include/TU/GaussianConvolver.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/IIRFilter.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Profiler.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/mmInstructions.h \
		/usr/local/include/TU/types.h \
		/usr/local/include/TU/utility.h
HDRS		=
SRCS		= main.cc
OBJS		= main.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.4 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/p.mk

###
main.o: /usr/local/include/TU/Image++.h /usr/local/include/TU/Geometry++.h \
	/usr/local/include/TU/utility.h /usr/local/include/TU/Vector++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/types.h \
	/usr/local/include/TU/Minimize.h /usr/local/include/TU/Profiler.h \
	../TU/CudaUtility.h ../TU/TU/CudaArray++.h \
	/usr/local/include/TU/GaussianConvolver.h \
	/usr/local/include/TU/IIRFilter.h \
	/usr/local/include/TU/mmInstructions.h
