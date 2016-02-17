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

INCDIRS		= -I. -I$(PREFIX)/include -I$(CUDAHOME)/include
CPPFLAGS	= -DNDEBUG #-DSSE3
CFLAGS		= -O
NVCCFLAGS	= -O -std=c++11 --relaxed-constexpr
CCFLAGS		= $(CFLAGS)

LIBS		=
ifneq ($(findstring darwin,$(OSTYPE)),)
  LIBS	       += -framework IOKit -framework CoreFoundation -framework CoreServices
endif

LINKER		= $(CXX)

BINDIR		= $(PREFIX)/bin
LIBDIR		= $(PREFIX)/lib
INCDIR		= $(PREFIX)/include

#########################
#  Macros set by mkmf	#
#########################
SUFFIX		= .cc:sC .cpp:sC .cu:sC
EXTHDRS		= /usr/local/include/TU/Array++.h \
		/usr/local/include/TU/functional.h \
		/usr/local/include/TU/iterator.h \
		/usr/local/include/TU/pair.h \
		/usr/local/include/TU/simd/allocator.h \
		/usr/local/include/TU/simd/arithmetic.h \
		/usr/local/include/TU/simd/arm/allocator.h \
		/usr/local/include/TU/simd/arm/arch.h \
		/usr/local/include/TU/simd/arm/arithmetic.h \
		/usr/local/include/TU/simd/arm/bit_shift.h \
		/usr/local/include/TU/simd/arm/cast.h \
		/usr/local/include/TU/simd/arm/compare.h \
		/usr/local/include/TU/simd/arm/cvt.h \
		/usr/local/include/TU/simd/arm/dup.h \
		/usr/local/include/TU/simd/arm/insert_extract.h \
		/usr/local/include/TU/simd/arm/load_store.h \
		/usr/local/include/TU/simd/arm/logical.h \
		/usr/local/include/TU/simd/arm/lookup.h \
		/usr/local/include/TU/simd/arm/select.h \
		/usr/local/include/TU/simd/arm/shift.h \
		/usr/local/include/TU/simd/arm/type_traits.h \
		/usr/local/include/TU/simd/arm/vec.h \
		/usr/local/include/TU/simd/arm/zero.h \
		/usr/local/include/TU/simd/bit_shift.h \
		/usr/local/include/TU/simd/cast.h \
		/usr/local/include/TU/simd/compare.h \
		/usr/local/include/TU/simd/config.h \
		/usr/local/include/TU/simd/cvt.h \
		/usr/local/include/TU/simd/cvtdown_iterator.h \
		/usr/local/include/TU/simd/cvtup_iterator.h \
		/usr/local/include/TU/simd/dup.h \
		/usr/local/include/TU/simd/insert_extract.h \
		/usr/local/include/TU/simd/load_iterator.h \
		/usr/local/include/TU/simd/load_store.h \
		/usr/local/include/TU/simd/logical.h \
		/usr/local/include/TU/simd/lookup.h \
		/usr/local/include/TU/simd/misc.h \
		/usr/local/include/TU/simd/row_vec_iterator.h \
		/usr/local/include/TU/simd/select.h \
		/usr/local/include/TU/simd/shift.h \
		/usr/local/include/TU/simd/shift_iterator.h \
		/usr/local/include/TU/simd/simd.h \
		/usr/local/include/TU/simd/store_iterator.h \
		/usr/local/include/TU/simd/transform.h \
		/usr/local/include/TU/simd/type_traits.h \
		/usr/local/include/TU/simd/vec.h \
		/usr/local/include/TU/simd/x86/allocator.h \
		/usr/local/include/TU/simd/x86/arch.h \
		/usr/local/include/TU/simd/x86/arithmetic.h \
		/usr/local/include/TU/simd/x86/bit_shift.h \
		/usr/local/include/TU/simd/x86/cast.h \
		/usr/local/include/TU/simd/x86/compare.h \
		/usr/local/include/TU/simd/x86/cvt.h \
		/usr/local/include/TU/simd/x86/dup.h \
		/usr/local/include/TU/simd/x86/insert_extract.h \
		/usr/local/include/TU/simd/x86/load_store.h \
		/usr/local/include/TU/simd/x86/logical.h \
		/usr/local/include/TU/simd/x86/logical_base.h \
		/usr/local/include/TU/simd/x86/lookup.h \
		/usr/local/include/TU/simd/x86/select.h \
		/usr/local/include/TU/simd/x86/shift.h \
		/usr/local/include/TU/simd/x86/shuffle.h \
		/usr/local/include/TU/simd/x86/svml.h \
		/usr/local/include/TU/simd/x86/type_traits.h \
		/usr/local/include/TU/simd/x86/unpack.h \
		/usr/local/include/TU/simd/x86/vec.h \
		/usr/local/include/TU/simd/x86/zero.h \
		/usr/local/include/TU/simd/zero.h \
		/usr/local/include/TU/tuple.h
HDRS		= TU/CudaArray++.h \
		TU/CudaFilter.h \
		TU/CudaGaussianConvolver.h \
		TU/CudaTexture.h \
		TU/CudaUtility.h
SRCS		= CudaFilter.cu \
		CudaGaussianConvolver.cc \
		cudaOp3x3.cu \
		cudaSubsample.cu \
		cudaSuppressNonExtrema3x3.cu
OBJS		= CudaFilter.o \
		CudaGaussianConvolver.o \
		cudaOp3x3.o \
		cudaSubsample.o \
		cudaSuppressNonExtrema3x3.o

#include $(PROJECT)/lib/rtc.mk		# modified: CPPFLAGS, LIBS
#include $(PROJECT)/lib/cnoid.mk	# modified: CPPFLAGS, LIBS, LIBDIR
include $(PROJECT)/lib/lib.mk		# added:    PUBHDRS TARGHDRS
include $(PROJECT)/lib/common.mk
###
CudaFilter.o: TU/CudaFilter.h TU/CudaArray++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/x86/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/x86/vec.h \
	/usr/local/include/TU/simd/x86/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/x86/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/x86/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/x86/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/x86/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/x86/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/x86/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/x86/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/dup.h /usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/x86/cvt.h \
	/usr/local/include/TU/simd/x86/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/x86/logical.h \
	/usr/local/include/TU/simd/x86/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/x86/dup.h \
	/usr/local/include/TU/simd/arm/dup.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/x86/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/x86/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/x86/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/x86/shuffle.h \
	/usr/local/include/TU/simd/x86/svml.h \
	/usr/local/include/TU/simd/transform.h \
	/usr/local/include/TU/simd/lookup.h \
	/usr/local/include/TU/simd/x86/lookup.h \
	/usr/local/include/TU/simd/arm/lookup.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h TU/CudaUtility.h
CudaGaussianConvolver.o: TU/CudaGaussianConvolver.h TU/CudaFilter.h \
	TU/CudaArray++.h /usr/local/include/TU/Array++.h \
	/usr/local/include/TU/iterator.h /usr/local/include/TU/tuple.h \
	/usr/local/include/TU/functional.h /usr/local/include/TU/pair.h \
	/usr/local/include/TU/simd/simd.h /usr/local/include/TU/simd/config.h \
	/usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/x86/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/x86/vec.h \
	/usr/local/include/TU/simd/x86/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/x86/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/x86/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/x86/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/x86/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/x86/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/x86/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/x86/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/dup.h /usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/x86/cvt.h \
	/usr/local/include/TU/simd/x86/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/x86/logical.h \
	/usr/local/include/TU/simd/x86/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/x86/dup.h \
	/usr/local/include/TU/simd/arm/dup.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/x86/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/x86/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/x86/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/x86/shuffle.h \
	/usr/local/include/TU/simd/x86/svml.h \
	/usr/local/include/TU/simd/transform.h \
	/usr/local/include/TU/simd/lookup.h \
	/usr/local/include/TU/simd/x86/lookup.h \
	/usr/local/include/TU/simd/arm/lookup.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h
cudaOp3x3.o: TU/CudaUtility.h TU/CudaArray++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/x86/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/x86/vec.h \
	/usr/local/include/TU/simd/x86/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/x86/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/x86/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/x86/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/x86/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/x86/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/x86/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/x86/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/dup.h /usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/x86/cvt.h \
	/usr/local/include/TU/simd/x86/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/x86/logical.h \
	/usr/local/include/TU/simd/x86/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/x86/dup.h \
	/usr/local/include/TU/simd/arm/dup.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/x86/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/x86/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/x86/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/x86/shuffle.h \
	/usr/local/include/TU/simd/x86/svml.h \
	/usr/local/include/TU/simd/transform.h \
	/usr/local/include/TU/simd/lookup.h \
	/usr/local/include/TU/simd/x86/lookup.h \
	/usr/local/include/TU/simd/arm/lookup.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h
cudaSubsample.o: TU/CudaUtility.h TU/CudaArray++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/x86/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/x86/vec.h \
	/usr/local/include/TU/simd/x86/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/x86/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/x86/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/x86/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/x86/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/x86/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/x86/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/x86/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/dup.h /usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/x86/cvt.h \
	/usr/local/include/TU/simd/x86/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/x86/logical.h \
	/usr/local/include/TU/simd/x86/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/x86/dup.h \
	/usr/local/include/TU/simd/arm/dup.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/x86/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/x86/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/x86/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/x86/shuffle.h \
	/usr/local/include/TU/simd/x86/svml.h \
	/usr/local/include/TU/simd/transform.h \
	/usr/local/include/TU/simd/lookup.h \
	/usr/local/include/TU/simd/x86/lookup.h \
	/usr/local/include/TU/simd/arm/lookup.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h
cudaSuppressNonExtrema3x3.o: TU/CudaUtility.h TU/CudaArray++.h \
	/usr/local/include/TU/Array++.h /usr/local/include/TU/iterator.h \
	/usr/local/include/TU/tuple.h /usr/local/include/TU/functional.h \
	/usr/local/include/TU/pair.h /usr/local/include/TU/simd/simd.h \
	/usr/local/include/TU/simd/config.h /usr/local/include/TU/simd/vec.h \
	/usr/local/include/TU/simd/type_traits.h \
	/usr/local/include/TU/simd/x86/type_traits.h \
	/usr/local/include/TU/simd/arm/type_traits.h \
	/usr/local/include/TU/simd/x86/vec.h \
	/usr/local/include/TU/simd/x86/arch.h \
	/usr/local/include/TU/simd/arm/vec.h \
	/usr/local/include/TU/simd/arm/arch.h \
	/usr/local/include/TU/simd/allocator.h \
	/usr/local/include/TU/simd/x86/allocator.h \
	/usr/local/include/TU/simd/arm/allocator.h \
	/usr/local/include/TU/simd/load_store.h \
	/usr/local/include/TU/simd/x86/load_store.h \
	/usr/local/include/TU/simd/arm/load_store.h \
	/usr/local/include/TU/simd/zero.h \
	/usr/local/include/TU/simd/x86/zero.h \
	/usr/local/include/TU/simd/arm/zero.h \
	/usr/local/include/TU/simd/cast.h \
	/usr/local/include/TU/simd/x86/cast.h \
	/usr/local/include/TU/simd/arm/cast.h \
	/usr/local/include/TU/simd/insert_extract.h \
	/usr/local/include/TU/simd/x86/insert_extract.h \
	/usr/local/include/TU/simd/arm/insert_extract.h \
	/usr/local/include/TU/simd/shift.h \
	/usr/local/include/TU/simd/x86/shift.h \
	/usr/local/include/TU/simd/arm/shift.h \
	/usr/local/include/TU/simd/bit_shift.h \
	/usr/local/include/TU/simd/x86/bit_shift.h \
	/usr/local/include/TU/simd/arm/bit_shift.h \
	/usr/local/include/TU/simd/dup.h /usr/local/include/TU/simd/cvt.h \
	/usr/local/include/TU/simd/x86/cvt.h \
	/usr/local/include/TU/simd/x86/unpack.h \
	/usr/local/include/TU/simd/arm/cvt.h \
	/usr/local/include/TU/simd/logical.h \
	/usr/local/include/TU/simd/x86/logical.h \
	/usr/local/include/TU/simd/x86/logical_base.h \
	/usr/local/include/TU/simd/arm/logical.h \
	/usr/local/include/TU/simd/x86/dup.h \
	/usr/local/include/TU/simd/arm/dup.h \
	/usr/local/include/TU/simd/compare.h \
	/usr/local/include/TU/simd/x86/compare.h \
	/usr/local/include/TU/simd/arm/compare.h \
	/usr/local/include/TU/simd/select.h \
	/usr/local/include/TU/simd/x86/select.h \
	/usr/local/include/TU/simd/arm/select.h \
	/usr/local/include/TU/simd/arithmetic.h \
	/usr/local/include/TU/simd/x86/arithmetic.h \
	/usr/local/include/TU/simd/arm/arithmetic.h \
	/usr/local/include/TU/simd/misc.h \
	/usr/local/include/TU/simd/x86/shuffle.h \
	/usr/local/include/TU/simd/x86/svml.h \
	/usr/local/include/TU/simd/transform.h \
	/usr/local/include/TU/simd/lookup.h \
	/usr/local/include/TU/simd/x86/lookup.h \
	/usr/local/include/TU/simd/arm/lookup.h \
	/usr/local/include/TU/simd/load_iterator.h \
	/usr/local/include/TU/simd/store_iterator.h \
	/usr/local/include/TU/simd/cvtdown_iterator.h \
	/usr/local/include/TU/simd/cvtup_iterator.h \
	/usr/local/include/TU/simd/shift_iterator.h \
	/usr/local/include/TU/simd/row_vec_iterator.h
