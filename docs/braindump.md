1. Quick and Dirty implementation
   - The goal here is to have something that takes EDGE as input and some user-defined tensors, and spits out fibertree results (both code and actual output tensor).
   - we should be able to extract the state of the EDGE system at any point in the iteration space for debug purposes
   - The primary goal of this version is for debugging EDGE Einsums
     - we also want to see what an EDGE system needs...
       
2. EDGE to MLIR
   - something that can generate MLIR so that we can target multiple backends given an Einsum.
   - here, we may be able to start adding optimizatin passes to things
3. EDGE to Egglog
   - this is for our algebraic passes.
