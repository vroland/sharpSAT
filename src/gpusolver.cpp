#include "gpusolver.h"
#include <iostream>

mpz_class GPUSolver::solveComponent(const Component* comp, AltComponentAnalyzer& analyzer) {
   std::cerr << "\ncache miss, gpu candidate:" << std::endl;
   std::cerr << "push back: " << comp->id() << std::endl
             << "num vars:" << comp->num_variables() << std::endl;

   // ids are component-local
   for (auto it = comp->varsBegin(); *it != varsSENTINEL; it++) {
       std::cerr << "var: " << *it << std::endl;

       bool truths[2] = {true, false};
       for (auto t : truths) {
           for (auto lit = (*literals)[LiteralID(*it, t)].binary_links_.begin(); *lit != SENTINEL_LIT; lit++) {
                (*lit).print();
                std::cerr << std::endl;
           }
       }
       for (auto t : truths) {
           for (auto ofs : (*literals)[LiteralID(*it, t)].watch_list_) {
               for (auto cit = lit_pool_->begin() + ofs; *cit != SENTINEL_CL; cit++) {
                    (*cit).print();
               }
               std::cerr << std::endl;
           }
       }
   }
   return mpz_class(0);
}
