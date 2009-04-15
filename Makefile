#
#  $Id: Makefile,v 1.1 2009-04-15 00:32:05 ueshiba Exp $
#
#################################
#  User customizable macros	#
#################################
DEST		= $(LIBDIR)
INCDIR		= $(HOME)/include/TU
INCDIRS		= -I. -I$(HOME)/include -I$(CUDASDK)/common/inc

NAME		= $(shell basename $(PWD))

CPPFLAGS	=
CFLAGS		= -g
NVCCFLAGS	= -g
ifeq ($(CCC), icpc)
  CFLAGS	= -O3
  NVCCFLAGS	= -O		# -O2以上にするとコンパイルエラーになる．
  ifeq ($(OSTYPE), darwin)
    CPPFLAGS   += -DSSE3
    CFLAGS     += -axP
  else
    CPPFLAGS   += -DSSSE3
    CFLAGS     += -xN
  endif
endif
CCFLAGS		= $(CFLAGS)

LINKER		= $(CCC)

#########################
#  Macros set by mkmf	#
#########################
.SUFFIXES:	.cu
SUFFIX		= .cc:sC .cu:sC
EXTHDRS		= TU/Cuda++.h
HDRS		= Cuda++.h \
		CudaDeviceMemory.h
SRCS		= Cuda.cu
OBJS		= Cuda.o

#########################
#  Macros used by RCS	#
#########################
REV		= $(shell echo $Revision: 1.1 $	|		\
		  sed 's/evision://'		|		\
		  awk -F"."					\
		  '{						\
		      for (count = 1; count < NF; count++)	\
			  printf("%d.", $$count);		\
		      printf("%d", $$count + 1);		\
		  }')

include $(PROJECT)/lib/l.mk
###
Cuda.o: TU/Cuda++.h
