/*
 * alt_component_analyzer.cpp
 *
 *  Created on: Mar 5, 2013
 *      Author: mthurley
 */


#include "alt_component_analyzer.h"
#include <bitset>
#include "gputypes.h"


extern unsigned long long componentModelCount(const std::vector<GPUClause>& clauses, uint64_t variable_count);

void AltComponentAnalyzer::initialize(LiteralIndexedVector<Literal> & literals,
    vector<LiteralID> &lit_pool) {

  max_variable_id_ = literals.end_lit().var() - 1;

  search_stack_.reserve(max_variable_id_ + 1);
  var_frequency_scores_.resize(max_variable_id_ + 1, 0);
  variable_occurrence_lists_pool_.clear();
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
    // BEGIN data for binary clauses
    variable_link_list_offsets_[v] = unified_variable_links_lists_pool_.size();
    for (auto l : literals[LiteralID(v, false)].binary_links_)
      if (l != SENTINEL_LIT)
        unified_variable_links_lists_pool_.push_back(l.var());

    for (auto l : literals[LiteralID(v, true)].binary_links_)
      if (l != SENTINEL_LIT)
        unified_variable_links_lists_pool_.push_back(l.var());

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
      if(manageSearchOccurrenceOf(LiteralID(*p,true))){
        var_frequency_scores_[*p]++;
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

int64_t AltComponentAnalyzer::solveComponentGPU(const Component* comp) {

   unsigned var_index = 0;
   vector<GPUClause> clauses;

   auto vars_end = comp->varsBegin();
   for (; *vars_end != varsSENTINEL; vars_end++) {};

   auto find_var_index = [&](VariableIndex var) -> int {
       // since variables seem to be ordered, binary search *could* be faster
       auto pos = find(comp->varsBegin(), vars_end,  var);
       // there cannot be an unknown variable, or the component is not disjoint
       //assert(pos != vars_end);
       if (pos == vars_end) {
           return -1;
       }
       return pos - comp->varsBegin();
   };

   for (auto it = comp->varsBegin(); it < vars_end; it++) {
       //std::cerr << "\nvar: " << *it << std::endl;

       bool truths[2] = {true, false};

       // variable is not assigned
       assert(isActive(*it));

       for (auto t : truths) {
           LiteralID self = LiteralID(*it, t);
           for (auto lit = (*literals_)[self].binary_links_.begin(); *lit != SENTINEL_LIT; lit++) {
               if (isActive(lit->var())) {
                   int other_index = find_var_index(lit->var());
                   if (other_index == -1) {
                        //cerr << "weird." << endl;
                        return -1;
                   }
                   assert(other_index >= 0);
                   GPUClause clause;
                   clause.vars |= 1 << var_index;
                   clause.vars |= 1 << other_index;
                   clause.neg_vars |= !t << var_index;
                   clause.neg_vars |= !lit->sign() << other_index;
                   clauses.push_back(clause);
               } else {
                   // all non-active binary clauses
                   // must be satisifed, or component is not sound as descibed in paper.
                   // (resolved pair variable would require self to be satisfied)
                   // This is done through BCP?
                   //FIXME: assert(isSatisfied(*lit));
               }
           }
           for (auto ofs = (*literals_)[self].watch_list_.rbegin(); *ofs != SENTINEL_CL; ofs++) {
               int active_vars = 0;
               GPUClause clause;
               clause.vars |= 1 << var_index;
               clause.neg_vars |= !t << var_index;
               for (auto cit = lit_pool_->begin() + *ofs; *cit != clsSENTINEL; cit++) {
                   if (isActive(cit->var())) {
                    int other_index = find_var_index(cit->var());
                    // FIXME: these are unassigned variables wich are not part of
                    // the component, but ignoring their clauses seems to yield
                    // correct results :O
                    if (other_index == -1) {
                        //cerr << "weird." << endl;
                        return -1;
                        //active_vars = -1;
                        continue;
                    }
                    clause.vars |= 1 << other_index;
                    clause.neg_vars |= !cit->sign() << other_index;
                    active_vars++;
                   // clause satisfied -> skip
                   } else if (isSatisfied(*cit)) {
                     active_vars = -1;
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
                   clauses.push_back(clause);
               }
           }
       }
       var_index++;
   }

   //cerr << "variables:" << var_index << endl;
   uint64_t unrestrained = 0;
   for (auto c : clauses) {
       unrestrained |= c.vars;
       //cout << std::bitset<32>(c.vars) << endl;
       //cout << std::bitset<32>(c.neg_vars) << endl << endl;
   }
   unsigned mc = componentModelCount(clauses, var_index);
   //cerr << "model count: " << mc << endl;
   return mc;
};
