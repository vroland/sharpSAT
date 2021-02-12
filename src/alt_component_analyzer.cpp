/*
 * alt_component_analyzer.cpp
 *
 *  Created on: Mar 5, 2013
 *      Author: mthurley
 */


#include "alt_component_analyzer.h"
#include <algorithm>
#include "gputypes.h"
#include <gpusat.h>
#include <chrono>
#include <FitnessFunctions/CutSetWidthFitnessFunction.h>


//extern unsigned long long componentModelCount(const std::vector<GPUClause>& clauses, uint64_t variable_count);

void AltComponentAnalyzer::initialize(LiteralIndexedVector<Literal> & literals,
    vector<LiteralID> &lit_pool) {

  max_variable_id_ = literals.end_lit().var() - 1;

  search_stack_.reserve(max_variable_id_ + 1);
  var_frequency_scores_.resize(max_variable_id_ + 1, 0);
  clause_offsets_.clear();
  variable_link_list_offsets_.clear();
  variable_link_list_offsets_.resize(max_variable_id_ + 1, 0);

  vector<vector<ClauseOfs> > occs(max_variable_id_ + 1);
  vector<vector<unsigned> > occ_long_clauses(max_variable_id_ + 1);
  vector<vector<unsigned> > occ_ternary_clauses(max_variable_id_ + 1);

  vector<unsigned> tmp;
  max_clause_id_ = 0;
  unsigned curr_clause_length = 0;
  auto it_curr_cl_st = lit_pool.begin();

  for (auto it_lit = lit_pool.begin(); it_lit < lit_pool.end(); it_lit++) {
    if (*it_lit == SENTINEL_LIT) {

      if (it_lit + 1 == lit_pool.end())
        break;

      max_clause_id_++;
      it_lit += ClauseHeader::overheadInLits();
      it_curr_cl_st = it_lit + 1;
      clause_offsets_.push_back((it_curr_cl_st - lit_pool.begin()));
      curr_clause_length = 0;

    } else {
      assert(it_lit->var() <= max_variable_id_);
      curr_clause_length++;

      getClause(tmp,it_curr_cl_st, *it_lit);

      assert(tmp.size() > 1);

      if(tmp.size() == 2) {
      //if(false){
        occ_ternary_clauses[it_lit->var()].push_back(max_clause_id_);
        occ_ternary_clauses[it_lit->var()].insert(occ_ternary_clauses[it_lit->var()].end(),
            tmp.begin(), tmp.end());
      } else {
        occs[it_lit->var()].push_back(max_clause_id_);
        occs[it_lit->var()].push_back(occ_long_clauses[it_lit->var()].size());
        occ_long_clauses[it_lit->var()].insert(occ_long_clauses[it_lit->var()].end(),
            tmp.begin(), tmp.end());
        occ_long_clauses[it_lit->var()].push_back(SENTINEL_LIT.raw());
      }

    }
  }

  ComponentArchetype::initArrays(max_variable_id_, max_clause_id_);
  // the unified link list
  unified_variable_links_lists_pool_.clear();
  unified_variable_links_lists_pool_.push_back(0);
  unified_variable_links_lists_pool_.push_back(0);
  for (unsigned v = 1; v < occs.size(); v++) {
    // BEGIN data for binary clauses of negative literal
    variable_link_list_offsets_[v] = unified_variable_links_lists_pool_.size();
    for (auto l : literals[LiteralID(v, false)].binary_links_)
      if (l != SENTINEL_LIT)
        unified_variable_links_lists_pool_.push_back(l.raw());

    unified_variable_links_lists_pool_.push_back(0);

    // BEGIN data for binary clauses of positive literal
    for (auto l : literals[LiteralID(v, true)].binary_links_)
      if (l != SENTINEL_LIT)
        unified_variable_links_lists_pool_.push_back(l.raw());

    unified_variable_links_lists_pool_.push_back(0);

    // BEGIN data for ternary clauses
    unified_variable_links_lists_pool_.insert(
        unified_variable_links_lists_pool_.end(),
        occ_ternary_clauses[v].begin(),
        occ_ternary_clauses[v].end());

    unified_variable_links_lists_pool_.push_back(0);

    // BEGIN data for long clauses
    for(auto it = occs[v].begin(); it != occs[v].end(); it+=2){
      unified_variable_links_lists_pool_.push_back(*it);
      unified_variable_links_lists_pool_.push_back(*(it + 1) +(occs[v].end() - it));
    }

    unified_variable_links_lists_pool_.push_back(0);

    unified_variable_links_lists_pool_.insert(
        unified_variable_links_lists_pool_.end(),
        occ_long_clauses[v].begin(),
        occ_long_clauses[v].end());
  }

  literals_ = &literals;
  lit_pool_ = &lit_pool;
}


//void AltComponentAnalyzer::recordComponentOf(const VariableIndex var) {
//
//  search_stack_.clear();
//  setSeenAndStoreInSearchStack(var);
//
//  for (auto vt = search_stack_.begin(); vt != search_stack_.end(); vt++) {
//    //BEGIN traverse binary clauses
//    assert(isActive(*vt));
//    unsigned *p = beginOfLinkList(*vt);
//    for (; *p; p++) {
//      if(isUnseenAndActive(*p)){
//        setSeenAndStoreInSearchStack(*p);
//        var_frequency_scores_[*p]++;
//        var_frequency_scores_[*vt]++;
//      }
//    }
//    //END traverse binary clauses
//    auto s = p;
//    for ( p++; *p ; p+=3) {
////      if(archetype_.clause_unseen_in_sup_comp(*p)){
////        LiteralID * pstart_cls = reinterpret_cast<LiteralID *>(p + 1);
////        searchThreeClause(*vt,*p, pstart_cls);
////      }
//    }
//    //END traverse ternary clauses
//
//    for (p++; *p ; p +=2) {
//      if(archetype_.clause_unseen_in_sup_comp(*p)){
//        LiteralID * pstart_cls = reinterpret_cast<LiteralID *>(p + 1 + *(p+1));
//        searchClause(*vt,*p, pstart_cls);
//      }
//    }
//
//    for ( s++; *s ; s+=3) {
//          if(archetype_.clause_unseen_in_sup_comp(*s)){
//            LiteralID * pstart_cls = reinterpret_cast<LiteralID *>(s + 1);
//            searchThreeClause(*vt,*s, pstart_cls);
//          }
//        }
//  }
//}

void AltComponentAnalyzer::recordComponentOf(const VariableIndex var) {

  search_stack_.clear();
  setSeenAndStoreInSearchStack(var);

  for (auto vt = search_stack_.begin(); vt != search_stack_.end(); vt++) {
    //BEGIN traverse binary clauses
    assert(isActive(*vt));
    unsigned *p = beginOfLinkList(*vt);
    for (; *p; p++) {
      auto lit = reinterpret_cast<const LiteralID *>(p);
      if(manageSearchOccurrenceOf(*lit)){
        var_frequency_scores_[lit->var()]++;
        var_frequency_scores_[*vt]++;
      }
    }
    for (p++; *p; p++) {
      auto lit = reinterpret_cast<const LiteralID *>(p);
      if(manageSearchOccurrenceOf(*lit)) {
        var_frequency_scores_[lit->var()]++;
        var_frequency_scores_[*vt]++;
      }
    }
    //END traverse binary clauses

    for ( p++; *p ; p+=3) {
      if(archetype_.clause_unseen_in_sup_comp(*p)){
        LiteralID litA = *reinterpret_cast<const LiteralID *>(p + 1);
        LiteralID litB = *(reinterpret_cast<const LiteralID *>(p + 1) + 1);
        if(isSatisfied(litA)|| isSatisfied(litB))
          archetype_.setClause_nil(*p);
        else {
          var_frequency_scores_[*vt]++;
          manageSearchOccurrenceAndScoreOf(litA);
          manageSearchOccurrenceAndScoreOf(litB);
          archetype_.setClause_seen(*p,isActive(litA) &
              isActive(litB));
        }
      }
    }
    //END traverse ternary clauses

    for (p++; *p ; p +=2)
      if(archetype_.clause_unseen_in_sup_comp(*p))
        searchClause(*vt,*p, reinterpret_cast<LiteralID *>(p + 1 + *(p+1)));

  }
}

mpz_class AltComponentAnalyzer::solveComponentGPU(const Component* comp) {

   unsigned var_index = 0;
   vector<GPUClause> clauses;

   auto long_clauses = 0;

   /*
   if (search_stack_.size() < 16 || search_stack_.size() > 22) {
     return -1;
   }
   */

   vector<VariableIndex> vstack(search_stack_);
   sort(vstack.begin(), vstack.end());

   gpusat::satformulaType formula;
   formula.numVars = vstack.size();

   auto local_var_index = [&](VariableIndex var) -> int {
       auto pos = lower_bound(vstack.begin(), vstack.end(),  var);
       // there cannot be an unknown variable, or the component is not disjoint
       //assert(pos != vars_end);
       if (*pos != var) {
           return -1;
       }
       // variables in htd start with one.
       return pos - vstack.begin() + 1;
   };
   auto mul_sign = [](LiteralID& lit) -> int64_t { return lit.sign() ? 1 : -1; };

   for (auto it = search_stack_.begin(); it < search_stack_.end(); it++) {
       //std::cerr << "\nvar: " << *it << std::endl;

       bool truths[2] = {true, false};

       // variable is not assigned
       assert(isActive(*it));
       for (auto t : truths) {
           LiteralID self = LiteralID(*it, t);
           int64_t self_local = local_var_index(self.var()) * mul_sign(self);
           for (auto lit = (*literals_)[self].binary_links_.begin(); *lit != SENTINEL_LIT; lit++) {
               if (isActive(lit->var())) {
                   int64_t local = local_var_index(lit->var());
                   if (local < 0) {
                       cout << "bin clause not in stack: " << lit->var() << endl;
                       return -1;
                   }
                   formula.clause_offsets.push_back(formula.clause_bag.size());
                   formula.clause_bag.push_back(self_local);
                   formula.clause_bag.push_back(local * mul_sign(*lit));
                   sort(formula.clause_bag.end() - 2, formula.clause_bag.end(), gpusat::compVars);
                   formula.clause_bag.push_back(0);
               } else {
                   // all non-active binary clauses
                   // must be satisifed, or component is not sound as descibed in paper.
                   // (resolved pair variable would require self to be satisfied)
                   // FIXME: This does not hold :/
                   //assert(local_var_index(lit->var()) == -1);
               }
           }
       }
       var_index++;
    }

    for (auto cid = comp->clsBegin(); *cid != clsSENTINEL; cid++) {
       int active_vars = 0;
       vector<int64_t> clause;
       ClauseOfs ofs = clauseIdToOfs(*cid);
       for (auto cit = lit_pool_->begin() + ofs; *cit != clsSENTINEL; cit++) {
           if (isActive(cit->var())) {
               int64_t local = local_var_index(cit->var());
               if (local < 0) {
                   cout << "long clause not in stack: " << cit->var() << endl;
                   return -1;
               }
               clause.push_back(local * mul_sign(*cit));
               active_vars++;
           // clause satisfied -> skip
           } else if (isSatisfied(*cit)) {
             //continue;
             active_vars = -1;
             cout << "clause satisfied" << endl;
             break;
           } else {
            // if clause is still active,
            // there should not be any satisfying literal.
            assert(isResolved(*cit));
           }
       }
       // clause not skipped
       if (active_vars > 0) {
           assert(active_vars >= 1);
           long_clauses++;
           sort(clause.begin(), clause.end(), gpusat::compVars);
           formula.clause_offsets.push_back(formula.clause_bag.size());
           formula.clause_bag.reserve(formula.clause_bag.size() + clause.size());
           formula.clause_bag.insert(formula.clause_bag.end(), clause.begin(), clause.end());
           formula.clause_bag.push_back(0);
       }
    }

   //cout << "p cnf " << formula.numVars << " " << formula.clauses.size() << endl;
   // sort clauses, this is expected by gpusat
   // for (auto& clause : formula.clauses) {
   //      sort(clause.begin(), clause.end(), gpusat::compVars);
   //      for (auto lit : clause) {
   //          //cout << lit << " ";
   //      }
   //      //cout << "0" << endl;
   // }

   auto fitness = new gpusat::CutSetWidthFitnessFunction();

   assert(formula.facts.empty());
   auto start = chrono::high_resolution_clock::now();
   auto decomp = gpusat::GPUSAT::decompose(formula, *fitness, 30);
   auto end = chrono::high_resolution_clock::now();

    cout << "PROF decomposition time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << endl;
   if (decomp.width >= 30) {
       cout << "too large treewidth." << endl;
       return -1;
   }
   auto result = gpusat::GPUSAT::preprocess(formula, decomp, gsat.recommended_bag_width());
   if (result.first != gpusat::PreprocessingResult::SUCCESS) {
        return -1;
   }
   //cout << "treewidth: " << decomp.width << endl;
   //cout << "clauses extracted: " << formula.clauses.size() << " clauses declared: " << comp->numLongClauses() << " long: " << long_clauses << "num vars: " << formula.numVars << endl;
   gpusat::GpusatConfig cfg;
   cfg.trace = false;
   cfg.solution_type = gpusat::dataStructure::ARRAY;
   cfg.solve_cfg.no_exponent = false;
   cfg.solve_cfg.weighted = false;
   cfg.max_bag_size = 0;

   boost::multiprecision::cpp_bin_float_100 mc;
   try {
       mc = gsat.solve(formula, decomp, cfg, result.second);
   } catch (const std::bad_optional_access) {
       return -1;
   }


   //cerr << "model count: " << mc.str() << endl;
   // FIXME: conversion via string is a crude hack
   return mpz_class(mc.str(100000));
};
