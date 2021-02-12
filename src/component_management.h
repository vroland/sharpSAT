/*
 * component_management.h
 *
 *  Created on: Aug 23, 2012
 *      Author: Marc Thurley
 */

#ifndef COMPONENT_MANAGEMENT_H_
#define COMPONENT_MANAGEMENT_H_



#include "component_types/component.h"
#include "component_cache.h"
#include "alt_component_analyzer.h"
//#include "component_analyzer.h"

#include <map>
#include <vector>
#include <optional>
#include "containers.h"
#include "stack.h"
#include "instance.h"

#include "solver_config.h"
using namespace std;

typedef AltComponentAnalyzer ComponentAnalyzer;

class ComponentManager {
public:
  ComponentManager(SolverConfiguration &config, DataAndStatistics &statistics,
        LiteralIndexedVector<TriValue> & lit_values, LiteralIndexedVector<vector<ClauseOfs> >& occurrence_lists_) :
        config_(config), statistics_(statistics), cache_(statistics),
        ana_(statistics,lit_values), occurrence_lists_(occurrence_lists_) {
  }

  ~ComponentManager() {
      cout << "component sizes:" << endl;
      for (auto it = component_stats.begin(); it != component_stats.end(); it++) {
        //cout << it->first << ": " << it->second << endl;
      }
  }

  void initialize(LiteralIndexedVector<Literal> & literals,
        vector<LiteralID> &lit_pool);

  unsigned scoreOf(VariableIndex v) {
      return ana_.scoreOf(v);
  }

  void cacheModelCountOf(unsigned stack_comp_id, const mpz_class &value) {
    if (config_.perform_component_caching) {
        auto component = component_stack_[stack_comp_id];
        cache_.storeValueOf(component->id(), value);
        if (component->cache_time_.has_value()) {
          auto now = chrono::high_resolution_clock::now();
          auto duration = chrono::duration_cast<chrono::microseconds>(now - component->cache_time_.value());
          cout << "PROF component duration: " << duration.count() << " vars: " << component->num_variables() << endl;
        }
    }
  }

  Component & superComponentOf(StackLevel &lev) {
    assert(component_stack_.size() > lev.super_component());
    return *component_stack_[lev.super_component()];
  }

  unsigned component_stack_size() {
    return component_stack_.size();
  }

  void cleanRemainingComponentsOf(StackLevel &top) {
    while (component_stack_.size() > top.remaining_components_ofs()) {
      if (cache_.hasEntry(component_stack_.back()->id()))
        cache_.entry(component_stack_.back()->id()).set_deletable();
      delete component_stack_.back();
      component_stack_.pop_back();
    }
    assert(top.remaining_components_ofs() <= component_stack_.size());
  }

  Component & currentRemainingComponentOf(StackLevel &top) {
    assert(component_stack_.size() > top.currentRemainingComponent());
    return *component_stack_[top.currentRemainingComponent()];
  }

  // checks for the next yet to explore remaining component of top
  // returns true if a non-trivial non-cached component
  // has been found and is now stack_.TOS_NextComp()
  // returns false if all components have been processed;
  inline bool findNextRemainingComponentOf(StackLevel &top);

  inline void recordRemainingCompsFor(StackLevel &top);

  inline void sortComponentStackRange(unsigned start, unsigned end);

  void gatherStatistics(){
//     statistics_.cache_bytes_memory_usage_ =
//	     cache_.recompute_bytes_memory_usage();
    cache_.compute_byte_size_infrasture();
  }

  void removeAllCachePollutionsOf(StackLevel &top);

private:

  SolverConfiguration &config_;
  DataAndStatistics &statistics_;

  vector<Component *> component_stack_;
  ComponentCache cache_;
  ComponentAnalyzer ana_;
  LiteralIndexedVector<vector<ClauseOfs> >& occurrence_lists_;
  map<unsigned, unsigned> component_stats;
};


void ComponentManager::sortComponentStackRange(unsigned start, unsigned end){
    assert(start <= end);
    // sort the remaining components for processing
    for (unsigned i = start; i < end; i++)
      for (unsigned j = i + 1; j < end; j++) {
        if (component_stack_[i]->num_variables()
            < component_stack_[j]->num_variables())
          swap(component_stack_[i], component_stack_[j]);
      }
  }

bool ComponentManager::findNextRemainingComponentOf(StackLevel &top) {
    // record Remaining Components if there are none!
    if (component_stack_.size() <= top.remaining_components_ofs())
      recordRemainingCompsFor(top);
    assert(!top.branch_found_unsat());
    if (top.hasUnprocessedComponents())
      return true;
    // if no component remains
    // make sure, at least that the current branch is considered SAT
    top.includeSolution(1);
    return false;
  }


void ComponentManager::recordRemainingCompsFor(StackLevel &top) {
   Component & super_comp = superComponentOf(top);
   unsigned new_comps_start_ofs = component_stack_.size();

   ana_.setupAnalysisContext(top, super_comp);

   int components_stored = 0;
   for (auto vt = super_comp.varsBegin(); *vt != varsSENTINEL; vt++)
     if (ana_.isUnseenAndActive(*vt) &&
         ana_.exploreRemainingCompOf(*vt)){

       Component *p_new_comp = ana_.makeComponentFromArcheType();
       CacheableComponent *packed_comp = new CacheableComponent(ana_.getArchetype().current_comp_for_caching_);
         if (!cache_.manageNewComponent(top, *packed_comp, true)){

            //component_stats[p_new_comp->num_variables()] += 1;
            components_stored++;
            p_new_comp->cache_time_ = nullopt;
            component_stack_.push_back(p_new_comp);
            p_new_comp->set_id(cache_.storeAsEntry(*packed_comp, super_comp.id()));

            // there may be a better place to do this?
            if (config_.use_gpusolve && p_new_comp->num_variables() >= 18 && p_new_comp->num_variables() <= 23) {

                float in_binary = 0.0;
                float in_long = 0.0;
                for (auto it = p_new_comp->varsBegin(); *it != varsSENTINEL; it++) {
                    //score1 += scoreOf(*it);

                    auto plit = (*ana_.literals_)[LiteralID(*it, true)];
                    auto nlit = (*ana_.literals_)[LiteralID(*it, false)];
                    for (auto l : plit.binary_links_) {
                        if (!ana_.isActive(l)) in_binary++;
                    }
                    for (auto l : nlit.binary_links_) {
                        if (!ana_.isActive(l)) in_binary++;
                    }
                    /*
                    for (auto cid = p_new_comp->clsBegin(); *cid != clsSENTINEL; cid++) {
                        auto ofs = ana_.clauseIdToOfs(*cid);
                        for (auto cit = ana_.lit_pool_->begin() + ofs; *cit != clsSENTINEL; cit++) {
                            if (!ana_.isActive(*cit)) in_long++;
                        }
                    }
                    */
                    //cout << "var: " << *it << " penalty: " << penalty << endl;
                }
                in_long = p_new_comp->numLongClauses();
                //score1 /= p_new_comp->num_variables();
                //connectedness_like /= p_new_comp->num_variables(); //* ana_.lit_pool_->size();
                //connectedness_like /= p_new_comp->num_variables(); //* ana_.lit_pool_->size();

                //cout << "gpu solve comp with " << p_new_comp->num_variables() << " vars, " << p_new_comp->numLongClauses() << " clauses. connectedness: " << connectedness_like << endl;
               //cout << "in binray: " << in_binary << " in long: " << in_long << endl;
               if (in_long > in_binary / 4) {

                   p_new_comp->cache_time_ = make_optional(chrono::high_resolution_clock::now());
                   auto model_count = ana_.solveComponentGPU(p_new_comp);
                   if (model_count > 0) {
                       assert(model_count >= 0);
                       //cout << "component depth: " << new_comps_start_ofs << " vars: " << p_new_comp->num_variables() << "model count: "
                       //     << model_count << " in_binary: " << in_binary <<  " in_long: " << in_long << endl;
                       cout << "PROF gpu-hint" << endl;
                       cacheModelCountOf(component_stack_.size() -1, model_count);
                       bool hit = cache_.manageNewComponent(top, *packed_comp, false);
                       assert(hit);
                       component_stack_.pop_back();
                       components_stored--;
                   } else {
                       cout << "invalid component!" << endl;
                   }
               }
            }
         }
         else {
           delete packed_comp;
           delete p_new_comp;
         }
     }

    if (components_stored > 0) {
        //component_stats[p_new_comp->num_variables()] += 1;
        component_stack_.back()->cache_time_ = make_optional(chrono::high_resolution_clock::now());
    }

   top.set_unprocessed_components_end(component_stack_.size());
   sortComponentStackRange(new_comps_start_ofs, component_stack_.size());
}

#endif /* COMPONENT_MANAGEMENT_H_ */
