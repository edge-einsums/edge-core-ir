# Terminology
### Let us begin with some definitions
- Map temp --> Reduce Temp --> Populate Temp/Final output
- reduce temp has the shape of the iteration space after all contracted rank variables have been removed (note that this is different from Map)

The populate action takes the following as input *at a given point in the reduced iteration space*:
  1. The reduce temp payload
  2. The reduce temp coordinate
  3. The current output fiber *for the mutable rank in consideration*
  4. A coordinate operator (user-defined)
  5. A compute operator (user-defined)

The populate action outputs:
  1. A modified output fiber *for the mutable rank in consideration*

Populate consists of two operators:
  1. The coordinate operator
  2. The compute operator

The coordinate operator takes as input:
  1. The reduce temp payload
  2. The reduce temp coordinate
  3. The current output fiber *for the mutable rank in consideration*

The coordinate operator outputs:
  1. A set of valid coordinates to be updated
  2. A set of dead coordinates to be deleted

  Note: the coordinate operator does NOT modify anything. We are not allowing populate to reduce.
  
  The populate action will process the output of coordinate and compute and perform the modifications.



The compute operator takes as input:
  1. A coordinate from the coordinate operator
  2. The reduce temp payload  
  Populate calls the compute operator for every valid coordinate from the coordinate operator.

The compute operator outputs:
   1. A (modified) payload. (per call)

Question:
  1. are we enforcing that populate MUST be at the innermost level of the loop nest?
    - leaf fiber --> implementation artifact
      - talking about a fiber does not require that that fiber be a leaf fiber
      - populate operates on a *mathematical* fiber where the payloads are always leaf payloads

EDGE 2 revisions (08/27/2024):
  - If we're using the default populate, the output is exactly the reduce temporary.
  - clarifying that the recipe works point-by-point
    - map, reduce, optional populate if reduction done (note we only iterate through once)
  -


  - why not allow the user to provide an initial state, modify it at will, etc..., but the final thing must be a valid fiber
    - this an implementation and doesn't followe the ODE.