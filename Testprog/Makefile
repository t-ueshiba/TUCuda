SUBDIR	= boxFilterTest		\
	  boxFilterTest2	\
	  eigen33Test		\
	  filterTest		\
	  fp16Test		\
	  gaussTest		\
	  guidedFilterTest	\
	  mappedMemoryTest	\
	  op3x3Test		\
	  cudaStereoTest	\
	  subsampleTest		\
	  testIIDCcuda		\
	  transposeTest		\
	  warpTest

TARGETS	= all clean depend

all:

$(TARGETS):
	@for d in $(SUBDIR); do				\
	  echo "";					\
	  echo "*** Current directory: $$d ***";	\
	  cd $$d;					\
	  $(MAKE) NAME=$$d $@;				\
	  cd ..;					\
	done
