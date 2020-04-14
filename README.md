# seed-reversal-gpu

Seed-reversal-gpu is the first step in the pipeline for generating a list of possible chunk seeds. This GitHub repository has a branch for each x alignment. The way it works is as follows:
- Feed the position of A2 (x0-x2 and x5-x15) (bottom left tree) or A1 (x3-x4) (bottom right tree) into a lattice to obtain a list of initial seed guesses. There is some weird manipulation to convert the GPU thread ID and offset into a point on the parallelogram, but it is otherwise the same concept.
- Filter out seeds where the height of that tree does not match what is seen on pack.png.
- Since this tree may not have generated at the start of tree generation, iterate over the possible numbers of calls between the start of tree generation and the generation of this tree. A tree attempt does 3 random calls when it fails and 19 when it is successful. Using an upper bound of 12 tree attempts, a list of possible number of calls is precomputed with dynamic programming.
- Checks whether this chunk generates small trees or big trees, and filters the seed out if it’s big trees.
- From the start of tree generation, simulate the tree generation forwards. A tree attempt fails if it’s not in the same location as in pack.png, and is successful if it is in the same location as in pack.png, this should find all matching seeds but can also generate false positives which will be filtered out by later programs. In the case of the blobby tree, any tree in that area is successful.
- On iterations where all trees we are looking for have been generated, we skip over the rest of the generation as fast as we can to the waterfall generation, and test whether any of the 50 waterfall attempts generate a waterfall in the correct location.
- Runs the LCG (RNG) backwards to the start of population to get the chunk seed.
