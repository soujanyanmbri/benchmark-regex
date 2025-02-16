# benchmark-regex

With SIMD (Single Instruction, Multiple Data), you can compare multiple characters at once. Instead of checking one character at a time, you “pack” a block of text into a wide register (say 128 bits = 16 characters) and perform a single instruction that compares all those characters simultaneously.



Key Parameter –  α: Here, α represents the number of characters that can be processed in parallel (for SSE, typically 𝛼 = 16 α=16). In many parts of the algorithms, a threshold  m0 = min{m,α/2} is used. This means that if the pattern is short, the algorithm uses the entire pattern; if the pattern is longer, it initially considers only the first half of a SIMD register’s width (i.e., up to 8 characters when using SSE) as a filter.

```
Text:   T[0]  T[1]  T[2]  T[3]  T[4]  T[5]  T[6]  T[7]  T[8]  ...
Pattern:    P[0]  P[1]  P[2]  P[3]

Naive Comparison (per position):
   For each position i in Text:
     Compare T[i] with P[0],
             T[i+1] with P[1],
             T[i+2] with P[2],
             T[i+3] with P[3]
     
SIMD Comparison:
   Load 16 characters from text into a SIMD register:
         [T[i], T[i+1], T[i+2], ..., T[i+15]]
   Broadcast P[0] into a SIMD register:
         [P[0], P[0], ..., P[0]]
   Compare all 16 elements in one instruction.
```


The EPSM family splits the problem into different cases based on the pattern length. We use a parameter α (alpha) that represents the number of characters processed in parallel—in our case, α = 16 (for 128-bit SSE).

### 1. EPSMa (EPSM-a): Filtering for Medium-Length Patterns

1. Define 𝑚0 = min(𝑚,𝛼/2)
    m0 =min(m,α/2). For example, if the pattern is 16 characters long, 𝑚0 = 8
2. For each of the first 𝑚0 characters in the pattern, “broadcast” the character into an entire SIMD register.
    Then, for each candidate position in the text, load a block of 16 characters and perform parallel comparisons using these broadcast vectors.
3. Break text block into multiple Tis,  T into multiple αs
```
Pattern p = [ P0, P1, P2, P3, P4, P5, P6, P7, ... ]
We set m0 = 8 (for m >= 8)

Broadcast each pattern character:
   B0 = [ P0, P0, P0, ... , P0 ]  (16 copies)
   B1 = [ P1, P1, P1, ... , P1 ]
    ...
   B7 = [ P7, P7, P7, ... , P7 ]

For each block in text:
   Text block T = [ T0, T1, T2, ... , T15 ]
   Compare:
      Compare T with B0 → get mask M0 (bit j = 1 if T[j] == P0)
      Compare T with B1 → get mask M1 (bit j = 1 if T[j] == P1)
       ...
      Compare T with B7 → get mask M7
   Intersect the masks (using bitwise AND) so that only positions where all 8 comparisons match remain.
   If a bit is set, then verify the full pattern.
```

### 2. EPSMb (EPSM-b): Short-Pattern Matching with Blending Idea: 

When the pattern is very short (e.g., 𝑚 ≤ 8 m≤8), the algorithm processes the text in blocks of 16 characters. It examines the block for a match in the first half of the register. To catch matches that might start near the end of a block, it “blends” adjacent blocks. That means it creates a new block by combining the upper half of the current block with the lower half of the next block.

```
Text Block T_i:  [ T0, T1, ... , T15 ]
Match positions:  Check positions 0 to (16-m0)
  (Example: m0 = 4, so check positions 0 to 12)

To catch boundary cases:
   Next block T_(i+1): [ T16, T17, ... , T31 ]
   Create blended block S:
       S = [ T8, T9, ... , T15, T16, T17, ... , T23 ]
   Then check S for the pattern.

```

Improvement:
Blending ensures that matches starting in the latter part of one block (which might otherwise be missed due to alignment restrictions) are still found, without incurring the overhead of shifting memory addresses one by one.


### 3 EPSMc (EPSM-c): Hash-Based Filtering for Medium-Length Patterns

Idea: For medium-length patterns (e.g., 𝑚 > 32 ), a direct comparison of all characters can be too costly. Instead, compute a fast hash (or fingerprint) on blocks of α characters from both the pattern and the text. Use a lookup (or filtering) step to identify candidate positions where the fingerprints match. Then verify the full pattern at those positions.


```
Pattern p (length m) -> Compute fingerprints for all substrings of length α.
  Example:
    p[0..15] → hash H0
    p[1..16] → hash H1
    ...
    p[m-α..m-1] → hash Hx

For each text block T_i (16 characters):
   Compute hash H for T_i.
   If H matches any fingerprint Hx, then candidate match exists.
   Verify full pattern.

```



```

         +------------------------+
         |  Pattern length (m)    |
         +------------------------+
                    |
         +----------+----------+
         |                     |
      m <= 8?             m > 8 ?
         |                     |
      Yes: Use EPSMb       +----------+
                           |          |
                      m <= 32?      m > 32?
                           |          |
                        Yes: Use   Yes: Use
                          EPSMa    EPSMc
```