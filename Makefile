TARGETS=Release Debug Profiling
#EXAMPLE_RUN_PARAMS?= ~/sharpSAT/test2.cnf
#EXAMPLE_RUN_PARAMS?= ~/instances/cachet/DQMR/or-60-10-5.cnf.wcnf
#EXAMPLE_RUN_PARAMS?= ~/instances/all_Count_combined/13A-1.cnf
EXAMPLE_RUN_PARAMS?= ~/instances/track1_all/track1_027.cnf
#EXAMPLE_RUN_PARAMS?= ~/instances/ai/hoos/Research/SAT/GenGCP/Flat200-479/flat200-99.cnf

.PHONY: configure_%
configure_%:
	mkdir -p build/$*
	(cd build/$* && cmake \
		-DCMAKE_BUILD_TYPE=$* \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-G Ninja \
	../../)

.PHONY: configure_Profiling
configure_Profiling:
	mkdir -p build/Profiling
	(cd build/Profiling && cmake \
		-DCMAKE_BUILD_TYPE=Profiling \
		-DCMAKE_CXX_FLAGS=-pg \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-G Ninja \
	../../)

.PHONY: clean_%
clean_%:
	rm -r build/$*

.PHONY: build_%
build_%: configure_%
	cmake --build build/$*

.PHONY: run_%
run_%: build_%
	build/$*/sharpSAT -gpu $(EXAMPLE_RUN_PARAMS)

.PHONY: run_nogpu_%
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
