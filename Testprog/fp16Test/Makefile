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

INCDIRS		= -I. -I../.. -I$(HOME)/src/TUTools++ -I$(CUDAHOME)/include 
CPPFLAGS	= -DNDEBUG
CFLAGS		= -g
NVCCFLAGS	= -g -G -arch=sm_53 -Xcompiler -mf16c
ifeq ($(shell arch), armv7l)
  CPPFLAGS     += -DNEON
else
  CPPFLAGS     += -DSSE4
endif
CCFLAGS		= $(CFLAGS)

LIBS		= -lTUTools++ #-lTUCuda++ -lintlc
LINKER		= $(NVCC)

BINDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		= ../../TU/cuda/Array++.h \
		../../TU/cuda/allocator.h \
		../../TU/cuda/fp16.h \
		/home/artuser/src/TUTools++/TU/Array++.h \
		/home/artuser/src/TUTools++/TU/algorithm.h \
		/home/artuser/src/TUTools++/TU/iterator.h \
		/home/artuser/src/TUTools++/TU/range.h \
		/home/artuser/src/TUTools++/TU/tuple.h \
		/home/artuser/src/TUTools++/TU/type_traits.h
HDRS		=
SRCS		= main.cu
OBJS		= main.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
main.o: ../../TU/cuda/Array++.h ../../TU/cuda/allocator.h \
	/home/artuser/src/TUTools++/TU/Array++.h \
	/home/artuser/src/TUTools++/TU/range.h \
	/home/artuser/src/TUTools++/TU/iterator.h \
	/home/artuser/src/TUTools++/TU/tuple.h \
	/home/artuser/src/TUTools++/TU/type_traits.h \
	/home/artuser/src/TUTools++/TU/algorithm.h ../../TU/cuda/fp16.h
