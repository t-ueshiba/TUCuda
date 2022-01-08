#
#  $Id$
#
#################################
#  User customizable macros	#
#################################
PROGRAM		= $(shell basename $(PWD))
#LIBRARY		= lib$(shell basename $(PWD))

VPATH		=

IDLS		=
MOCHDRS		=

INCDIRS		= -I../.. -I$(HOME)/src/TUTools++ -I$(CUDAHOME)/include
CPPFLAGS	= #-DNDEBUG -DUSE_TBB
CFLAGS		= -O3
NVCCFLAGS	= -O3 --extended-lambda #-arch=sm_53 -Xcompiler -mf16c
ifeq ($(shell arch), armv7l)
  CPPFLAGS     += -DNEON
else
  CPPFLAGS     += -DSSE3
endif
CCFLAGS		= $(CFLAGS)

LIBS		= -lTUCuda -lTUTools -ltbb
LINKER		= $(NVCC)

BINDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		= ../../TU/cuda/Array++.h \
		../../TU/cuda/BoxFilter.h \
		../../TU/cuda/GuidedFilter.h \
		../../TU/cuda/algorithm.h \
		../../TU/cuda/allocator.h \
		../../TU/cuda/chrono.h \
		../../TU/cuda/iterator.h \
		../../TU/cuda/tuple.h \
		../../TU/cuda/vec.h
HDRS		=
SRCS		= cpuJob.cc \
		cudaJob.cu \
		main.cu
OBJS		= cpuJob.o \
		cudaJob.o \
		main.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
cudaJob.o: ../../TU/cuda/chrono.h ../../TU/cuda/GuidedFilter.h \
	../../TU/cuda/tuple.h ../../TU/cuda/vec.h ../../TU/cuda/BoxFilter.h \
	../../TU/cuda/Array++.h ../../TU/cuda/allocator.h \
	../../TU/cuda/iterator.h ../../TU/cuda/algorithm.h
