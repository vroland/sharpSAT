TARGETS=Release Debug Profiling
#EXAMPLE_RUN_PARAMS?= ~/sharpSAT/test.cnf
EXAMPLE_RUN_PARAMS?= ~/instances/track1_all/track1_009.cnf
#EXAMPLE_RUN_PARAMS ?= ~/instances/cachet/DQMR/or-60-10-5.cnf.wcnf
#EXAMPLE_RUN_PARAMS?= ~/instances/ai/hoos/Research/SAT/GenGCP/Flat200-479/flat200-99.cnf

.PHONY: configure_%
configure_%:
	mkdir -p build/$*
	(cd build/$* && cmake \
		-DCMAKE_BUILD_TYPE=$* \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	../../)

build/%/Makefile: configure_%
	echo "configuring $*"

.PHONY: clean_%
clean_%:
	rm -r build/$*

.PHONY: build_%
build_%: build/%/Makefile
	(cd build/$* && make)

.PHONY: run_%
run_%: build_%
	build/$*/sharpSAT -gpu $(EXAMPLE_RUN_PARAMS)

.PHONY: run_%
run_nogpu_%: build_%
	build/$*/sharpSAT $(EXAMPLE_RUN_PARAMS)

.PHONY: prof_%
prof_%: build_%
	nvprof ./build/$*/sharpSAT -gpu $(EXAMPLE_RUN_PARAMS)

.PHONY: clean
clean: $(foreach target,$(TARGETS),clean_$(target))
	rm -r build/

.PHONY: build_all
build_all: $(foreach target,$(TARGETS),build_$(target))

.PHONY: configure
configure: $(foreach target,$(TARGETS),configure_$(target))
