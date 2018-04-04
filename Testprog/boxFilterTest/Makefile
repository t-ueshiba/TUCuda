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

INCDIRS		= -I../.. -I$(PREFIX)/include -I$(CUDAHOME)/include
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
		../../TU/cuda/algorithm.h \
		../../TU/cuda/allocator.h \
		../../TU/cuda/chrono.h \
		../../TU/cuda/fp16.h \
		../../TU/cuda/vec.h \
		/usr/local/include/TU/Array++.h \
		/usr/local/include/TU/BoxFilter.h \
		/usr/local/include/TU/Camera++.h \
		/usr/local/include/TU/Filter2.h \
		/usr/local/include/TU/Geometry++.h \
		/usr/local/include/TU/Image++.h \
		/usr/local/include/TU/Manip.h \
		/usr/local/include/TU/Minimize.h \
		/usr/local/include/TU/Profiler.h \
		/usr/local/include/TU/Vector++.h \
		/usr/local/include/TU/algorithm.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/pair.h \
		/usr/local/include/TU/range.h \
		/usr/local/include/TU/tuple.h
HDRS		= TU/cuda/NewBoxFilter.h \
		TU/cuda/NonRecursiveBoxFilter.h
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
cpuJob.o: /usr/local/include/TU/Array++.h /usr/local/include/TU/range.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/algorithm.h /usr/local/include/TU/BoxFilter.h \
	/usr/local/include/TU/Filter2.h /usr/local/include/TU/Profiler.h
cudaJob.o: /usr/local/include/TU/Image++.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/Manip.h /usr/local/include/TU/Camera++.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Minimize.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/range.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/algorithm.h \
	/usr/local/include/TU/Profiler.h ../../TU/cuda/chrono.h \
	../../TU/cuda/BoxFilter.h ../../TU/cuda/Array++.h \
	../../TU/cuda/allocator.h ../../TU/cuda/algorithm.h \
	TU/cuda/NonRecursiveBoxFilter.h TU/cuda/NewBoxFilter.h \
	../../TU/cuda/fp16.h ../../TU/cuda/vec.h
main.o: /usr/local/include/TU/Image++.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/Manip.h /usr/local/include/TU/Camera++.h \
	/usr/local/include/TU/Geometry++.h /usr/local/include/TU/Minimize.h \
	/usr/local/include/TU/Vector++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/range.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/algorithm.h \
	/usr/local/include/TU/BoxFilter.h /usr/local/include/TU/Filter2.h \
	/usr/local/include/TU/Profiler.h
