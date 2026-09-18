EDGE IR for other frontends and backends to hook into.



## My Miscellaneous Notes
- I now allow rank variable expressions as inputs to other rank variable expressions. This is because we should be able to accept both rank variables AND rank constants as inputs. Allowing other rank functions and rank arithmetic may be doing too much, but I'd rather be flexible than constrained. This does mean that we now have indirection in Einsums...
    - TODO: decide if we want to constrain it to just RankVariables (bare ones) and RankConstants (literal or shape-symbol)

2. Currently, I reject rank variable expressions/functions that don't hve a rank variable as input. Do we want this? It seems as though we may allow functions that don't take anything as input becuas ethen they can just produce a random output or a constant output. BUT the internal function can just ignore the input, so I don't think it is hurting anyone to force the rank variable functions to have a rank variable as input. 

3. If the user does the rank list on map/reduce, then the validator should provide hints/warnings if how they used it isn't matching. See 1. below in the Error Checking header. 

## Validator Notes
1. Overlaps need to be rejected in case statements. 


## Hints/Error checking
1. We should probably find a way to provide hints to the user if it seems like they want to reduce in some way, but their Einsum is malformed. Like adding v to the rank list, but then v appears in the output --> that's malformed. 
2. TODO (ignoring for now): the reduce-set derivation (reduced ranks = iteration ranks minus output ranks) is only valid for BASIC reduce ops. It's not always derivable -- an opaque output RVE like a user function f(a,w) can map iteration points to output coords in a way we can't figure out from structure. So the rank_list cross-check / derived-set hint above only applies to basic reduces. Skipping this validation for now; revisit when we handle opaque output RVEs. 

## Parser/Upstream Notes
1. 