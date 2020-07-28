TARGETS=Release Debug Profiling
EXAMPLE_RUN_PARAMS?= ~/GPUSAT_orig/examples/flat30-1.cnf

.PHONY: configure_%
configure_%:
	mkdir -p build/$*
	(cd build/$* && cmake \
		-DCMAKE_BUILD_TYPE=$* \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	../../)

.PHONY: clean_%
clean_%:
	rm -r build/$*

.PHONY: build_%
build_%:
	(cd build/$* && make)

.PHONY: run_%
run_%: build_%
	build/$*/sharpSAT $(EXAMPLE_RUN_PARAMS)

.PHONY: clean
clean: $(foreach target,$(TARGETS),clean_$(target))
	rm -r build/

.PHONY: build_all
build_all: $(foreach target,$(TARGETS),build_$(target))

.PHONY: configure
configure: $(foreach target,$(TARGETS),configure_$(target))
