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
CPPFLAGS	= -DNDEBUG -DUSE_TBB
CFLAGS		= -O3
NVCCFLAGS	= -O3 #-arch=sm_53 -Xcompiler -mf16c
ifeq ($(shell arch), armv7l)
  CPPFLAGS     += -DNEON
else
  CPPFLAGS     += -DSSE3
endif
CCFLAGS		= $(CFLAGS)

LIBS		= -lTUTools++ -lTUCuda++ -ltbb
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
		../../TU/cuda/vec.h \
		/home/ueshiba/src/TUTools++/TU/Array++.h \
		/home/ueshiba/src/TUTools++/TU/Camera++.h \
		/home/ueshiba/src/TUTools++/TU/Geometry++.h \
		/home/ueshiba/src/TUTools++/TU/Image++.h \
		/home/ueshiba/src/TUTools++/TU/Manip.h \
		/home/ueshiba/src/TUTools++/TU/Minimize.h \
		/home/ueshiba/src/TUTools++/TU/Profiler.h \
		/home/ueshiba/src/TUTools++/TU/Vector++.h \
		/home/ueshiba/src/TUTools++/TU/algorithm.h \
		/home/ueshiba/src/TUTools++/TU/iterator.h \
		/home/ueshiba/src/TUTools++/TU/pair.h \
		/home/ueshiba/src/TUTools++/TU/range.h \
		/home/ueshiba/src/TUTools++/TU/tuple.h
HDRS		=
SRCS		= cudaJob.cu \
		main.cu
OBJS		= cudaJob.o \
		main.o

#include $(PROJECT)/lib/rtc.mk		# IDLHDRS, IDLSRCS, CPPFLAGS, OBJS, LIBS
#include $(PROJECT)/lib/qt.mk		# MOCSRCS, OBJS
#include $(PROJECT)/lib/cnoid.mk	# CPPFLAGS, LIBS, LIBDIR
#include $(PROJECT)/lib/lib.mk		# PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
cudaJob.o: /home/ueshiba/src/TUTools++/TU/Profiler.h \
	../../TU/cuda/chrono.h ../../TU/cuda/GuidedFilter.h \
	../../TU/cuda/iterator.h ../../TU/cuda/vec.h \
	/home/ueshiba/src/TUTools++/TU/Image++.h \
	/home/ueshiba/src/TUTools++/TU/pair.h \
	/home/ueshiba/src/TUTools++/TU/Manip.h \
	/home/ueshiba/src/TUTools++/TU/Camera++.h \
	/home/ueshiba/src/TUTools++/TU/Geometry++.h \
	/home/ueshiba/src/TUTools++/TU/Minimize.h \
	/home/ueshiba/src/TUTools++/TU/Vector++.h \
	/home/ueshiba/src/TUTools++/TU/Array++.h \
	/home/ueshiba/src/TUTools++/TU/range.h \
	/home/ueshiba/src/TUTools++/TU/iterator.h \
	/home/ueshiba/src/TUTools++/TU/tuple.h \
	/home/ueshiba/src/TUTools++/TU/algorithm.h ../../TU/cuda/tuple.h \
	../../TU/cuda/BoxFilter.h ../../TU/cuda/Array++.h \
	../../TU/cuda/allocator.h ../../TU/cuda/algorithm.h
main.o: /home/ueshiba/src/TUTools++/TU/Image++.h \
	/home/ueshiba/src/TUTools++/TU/pair.h \
	/home/ueshiba/src/TUTools++/TU/Manip.h \
	/home/ueshiba/src/TUTools++/TU/Camera++.h \
	/home/ueshiba/src/TUTools++/TU/Geometry++.h \
	/home/ueshiba/src/TUTools++/TU/Minimize.h \
	/home/ueshiba/src/TUTools++/TU/Vector++.h \
	/home/ueshiba/src/TUTools++/TU/Array++.h \
	/home/ueshiba/src/TUTools++/TU/range.h \
	/home/ueshiba/src/TUTools++/TU/iterator.h \
	/home/ueshiba/src/TUTools++/TU/tuple.h \
	/home/ueshiba/src/TUTools++/TU/algorithm.h ../../TU/cuda/tuple.h \
	../../TU/cuda/iterator.h
