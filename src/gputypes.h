#ifndef GPU_TYPES_H
#define GPU_TYPES_H

typedef struct {
    // the nth least significant bit is set if the nth variable
    // is present in the clause
    uint32_t vars = 0;
    // the nth least significant bit is set if the nth variable
    // is negated in the clause
    uint32_t neg_vars = 0;
} GPUClause;

#endif /* GPU_TYPES_H */
