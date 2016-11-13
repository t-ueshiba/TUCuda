#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
#PROGRAM		= $(shell basename $(PWD))
LIBRARY		= lib$(shell basename $(PWD))

IDLDIR		= .
IDLS		=

INCDIRS		= -I. -I$(HOME)/src/TUTools++ -I$(PREFIX)/include -I$(CUDAHOME)/include
CPPFLAGS	= -DNDEBUG #-DSSE3
CFLAGS		= -O3
NVCCFLAGS	= -O3
CCFLAGS		= $(CFLAGS)

LIBS		=
ifneq ($(findstring darwin,$(OSTYPE)),)
  LIBS	       += -framework IOKit -framework CoreFoundation -framework CoreServices
endif

LINKER		= $(NVCC)

BINDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		= /Users/ueshiba/src/TUTools++/TU/Array++.h \
		/Users/ueshiba/src/TUTools++/TU/functional.h \
		/Users/ueshiba/src/TUTools++/TU/iterator.h \
		/Users/ueshiba/src/TUTools++/TU/pair.h \
		/Users/ueshiba/src/TUTools++/TU/tuple.h
HDRS		= TU/cuda/Array++.h \
		TU/cuda/BoxFilter.h \
		TU/cuda/FIRFilter.h \
		TU/cuda/FIRGaussianConvolver.h \
		TU/cuda/Texture.h \
		TU/cuda/algorithm.h \
		TU/cuda/allocator.h \
		TU/cuda/chrono.h \
		TU/cuda/functional.h
SRCS		= FIRFilter.cu \
		FIRGaussianConvolver.cc
OBJS		= FIRFilter.o \
		FIRGaussianConvolver.o

#include $(PROJECT)/lib/rtc.mk		# modified: CPPFLAGS, LIBS
#include $(PROJECT)/lib/cnoid.mk	# modified: CPPFLAGS, LIBS, LIBDIR
include $(PROJECT)/lib/lib.mk		# added:    PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
FIRFilter.o: TU/cuda/FIRFilter.h TU/cuda/Array++.h TU/cuda/allocator.h \
	/Users/ueshiba/src/TUTools++/TU/Array++.h \
	/Users/ueshiba/src/TUTools++/TU/iterator.h \
	/Users/ueshiba/src/TUTools++/TU/tuple.h \
	/Users/ueshiba/src/TUTools++/TU/functional.h \
	/Users/ueshiba/src/TUTools++/TU/pair.h TU/cuda/algorithm.h
FIRGaussianConvolver.o: TU/cuda/FIRGaussianConvolver.h TU/cuda/FIRFilter.h \
	TU/cuda/Array++.h TU/cuda/allocator.h \
	/home/ueshiba/src/TUTools++/TU/Array++.h \
	/home/ueshiba/src/TUTools++/TU/iterator.h \
	/home/ueshiba/src/TUTools++/TU/tuple.h \
	/home/ueshiba/src/TUTools++/TU/functional.h \
	/home/ueshiba/src/TUTools++/TU/pair.h TU/cuda/algorithm.h
TUCuda++.inst.o: TU/cuda/Array++.h TU/cuda/allocator.h \
	/home/ueshiba/src/TUTools++/TU/Array++.h \
	/home/ueshiba/src/TUTools++/TU/iterator.h \
	/home/ueshiba/src/TUTools++/TU/tuple.h \
	/home/ueshiba/src/TUTools++/TU/functional.h \
	/home/ueshiba/src/TUTools++/TU/pair.h TU/cuda/algorithm.h
chrono.o: TU/cuda/chrono.h
