#ifndef GPU_SOLVER_H
#define GPU_SOLVER_H

#include "component_types/component.h"
#include <gmpxx.h>
#include "instance.h"
#include "alt_component_analyzer.h"

class GPUSolver {
    public:
        mpz_class solveComponent(const Component* comp, AltComponentAnalyzer& analyzer);

        void initialize(LiteralIndexedVector<Literal>& literals, vector<LiteralID> &lit_pool) {

            this->literals = &literals;
            this->lit_pool_ = &lit_pool;
        };
        LiteralIndexedVector<Literal>* literals;
        vector<LiteralID>* lit_pool_;
};

#endif /* GPU_SOLVER_H */
