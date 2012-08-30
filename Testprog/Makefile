SUBDIR	= extremaTest		\
	  filterTest		\
	  gaussTest		\
	  op3x3Test		\
	  subsampleTest

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
