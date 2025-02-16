#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__x86_64__) || defined(_M_X64)
    #define ARCH "x86_64"
#elif defined(__i386__) || defined(_M_IX86)
    #define ARCH "x86 (32-bit)"
#elif defined(__aarch64__)
    #define ARCH "ARM64"
#elif defined(__arm__)
    #define ARCH "ARM"
#else
    #define ARCH "Unknown"
#endif

// Print the architecture.
void print_architecture() {
    printf("Architecture: %s\n", ARCH);
}

//
// Function pointer type for search routines.
//
typedef int (*search_func_t)(const char *text, int n, const char *pattern, int m);

//
// Naive search implementation (fallback).
//
int naive_search(const char *text, int n, const char *pattern, int m) {
    for (int i = 0; i <= n - m; i++) {
        if (memcmp(text + i, pattern, m) == 0)
            return i;
    }
    return -1;
}

//
// ----- x86 SSE4.2 implementations -----
// We assume α = 16 (i.e., 16 characters per SIMD register).
//
#if defined(__x86_64__) || defined(_M_X86) || defined(__i386__) || defined(_M_IX86)

#include <immintrin.h>
#include <cpuid.h>

// A simple EPSMa-like variant.
// It uses the first m0 = min(m, 8) characters to filter candidate positions.
int epsm_a(const char *text, int n, const char *pattern, int m) {
    int m0 = (m < 8 ? m : 8);
    // Precompute broadcast vectors for the first m0 characters.
    __m128i B[8];
    for (int i = 0; i < m0; i++) {
        B[i] = _mm_set1_epi8(pattern[i]);
    }
    // Slide over the text.
    for (int i = 0; i <= n - m; i++) {
        int candidate = 1;
        __m128i block = _mm_loadu_si128((const __m128i*)(text + i));
        for (int j = 0; j < m0; j++) {
            __m128i cmp = _mm_cmpeq_epi8(block, B[j]);
            int mask = _mm_movemask_epi8(cmp);
            // Check if the j-th bit in the mask is set.
            if (!(mask & (1 << j))) {
                candidate = 0;
                break;
            }
        }
        if (candidate) {
            // Verify full pattern if necessary.
            if (m == m0 || memcmp(text + i, pattern, m) == 0)
                return i;
        }
    }
    return -1;
}

// A short-pattern variant (EPSMb-like) that processes blocks of 16 bytes and blends adjacent blocks.
int epsm_b(const char *text, int n, const char *pattern, int m) {
    int m0 = (m < 8 ? m : 8);
    // Process text in chunks of 16.
    for (int i = 0; i <= n - m; i += 16) {
        __m128i block = _mm_loadu_si128((const __m128i*)(text + i));
        // Check positions within this block.
        for (int j = 0; j <= 16 - m0; j++) {
            if (i + j > n - m)
                break;
            if (memcmp(text + i + j, pattern, m0) == 0) {
                if (m == m0 || memcmp(text + i + j, pattern, m) == 0)
                    return i + j;
            }
        }
        // Blend with the next block to catch occurrences crossing the boundary.
        if (i + 16 < n) {
            __m128i block_next = _mm_loadu_si128((const __m128i*)(text + i + 16));
            __m128i blend = _mm_or_si128(_mm_srli_si128(block, 8), _mm_slli_si128(block_next, 8));
            char blend_buf[16];
            _mm_storeu_si128((__m128i*)blend_buf, blend);
            for (int j = 0; j <= 8 - m0; j++) {
                if (i + 8 + j > n - m)
                    break;
                if (memcmp(blend_buf + j, pattern, m0) == 0) {
                    if (m == m0 || memcmp(text + i + 8 + j, pattern, m) == 0)
                        return i + 8 + j;
                }
            }
        }
    }
    return -1;
}

// Unified epsm_search for x86 chooses between epsm_b and epsm_a based on pattern length.
int epsm_search_x86(const char *text, int n, const char *pattern, int m) {
    static search_func_t search_func = NULL;
    static int last_m = 0;
    if (search_func == NULL || last_m != m) {
        if (m <= 8)
            search_func = epsm_b;  // For very short patterns, use EPSMb-like.
        else if (m <= 32)
            search_func = epsm_a;  // For medium-length patterns, use EPSMa-like.
        else
            search_func = naive_search; // For long patterns, fallback.
        last_m = m;
    }
    return search_func(text, n, pattern, m);
}

#define epsm_search epsm_search_x86

//
// ----- ARM NEON implementation -----
//
#elif defined(__aarch64__) || defined(__arm__)

#include <arm_neon.h>
#include <stdint.h>

// A simple NEON-based implementation for short patterns (up to 16 bytes).
int epsm_neon(const char *text, int n, const char *pattern, int m) {
    if (m > 16) return -1;  // Out-of-scope.
    uint8_t patbuf[16] = {0};
    memcpy(patbuf, pattern, m);
    uint8x16_t pat = vld1q_u8(patbuf);
    for (int i = 0; i <= n - m; i++) {
        uint8x16_t block = vld1q_u8((const uint8_t*)(text + i));
        uint8x16_t eq = vceqq_u8(block, pat);
        uint8_t res[16];
        vst1q_u8(res, eq);
        int match = 1;
        for (int j = 0; j < m; j++) {
            if (res[j] != 0xFF) {
                match = 0;
                break;
            }
        }
        if (match)
            return i;
    }
    return -1;
}
int epsm_search(const char *text, int n, const char *pattern, int m) {
    return epsm_neon(text, n, pattern, m);
}
#else
// Fallback for other architectures.
int epsm_search(const char *text, int n, const char *pattern, int m) {
    return naive_search(text, n, pattern, m);
}
#endif

//
// Helper: Generate a text buffer by repeating a sample string until size is reached.
//
void generate_text(char *buffer, int size, const char *sample) {
    int sample_len = (int)strlen(sample);
    int pos = 0;
    while (pos < size) {
        int copy_len = sample_len;
        if (pos + copy_len > size)
            copy_len = size - pos;
        memcpy(buffer + pos, sample, copy_len);
        pos += copy_len;
    }
    buffer[size] = '\0';
}

//
// Main function: Benchmark different pattern sizes on a fixed large text.
//
int main() {
    print_architecture();

    // Use a large text (e.g., 1,000,000 characters).
    int textSize = 1000000;
    char *text = (char *)malloc(textSize + 1);
    if (!text) {
        fprintf(stderr, "Failed to allocate memory for the text.\n");
        return 1;
    }
    const char *sampleText = "This is a sample text for testing the EPSM algorithm. "
                             "We want to see if the pattern is found using SIMD! ";
    generate_text(text, textSize, sampleText);
    
    // Define an array of pattern lengths to test.
    int patLens[] = {4, 8, 16, 32, 64};
    int numPatLens = sizeof(patLens) / sizeof(patLens[0]);
    
    printf("Text size: %d characters\n", textSize);
    printf("-------------------------------------------------\n");
    
    // For each pattern length, extract a pattern from a fixed offset so a match is known.
    for (int p = 0; p < numPatLens; p++) {
        int m = patLens[p];
        // Ensure we have enough text.
        if (m > textSize) break;
        // For this POC, take a pattern starting at offset 50.
        const char *pattern = text + 50;
        // Ensure we compare exactly m characters.
        char *patternBuf = (char *)malloc(m + 1);
        if (!patternBuf) continue;
        memcpy(patternBuf, pattern, m);
        patternBuf[m] = '\0';
        
        // Display the pattern.
        printf("Pattern (length %d): \"%s\"\n", m, patternBuf);
        
        // Verify that both methods find a match.
        int pos1 = epsm_search(text, textSize, patternBuf, m);
        int pos2 = naive_search(text, textSize, patternBuf, m);
        if (pos1 >= 0)
            printf("EPSM: Pattern found at position %d\n", pos1);
        else
            printf("EPSM: Pattern not found.\n");
        if (pos2 >= 0)
            printf("Naive: Pattern found at position %d\n", pos2);
        else
            printf("Naive: Pattern not found.\n");
        
        // Choose a suitable iteration count (fewer iterations for longer patterns).
        int iterations = (m <= 8 ? 100000 : (m <= 32 ? 10000 : 1000));
        
        clock_t start, end;
        double epsm_time, naive_time;
        
        // Benchmark EPSM search.
        start = clock();
        for (int i = 0; i < iterations; i++) {
            volatile int dummy = epsm_search(text, textSize, patternBuf, m);
            (void)dummy;
        }
        end = clock();
        epsm_time = (double)(end - start) / CLOCKS_PER_SEC;
        
        // Benchmark naive search.
        start = clock();
        for (int i = 0; i < iterations; i++) {
            volatile int dummy = naive_search(text, textSize, patternBuf, m);
            (void)dummy;
        }
        end = clock();
        naive_time = (double)(end - start) / CLOCKS_PER_SEC;
        
        printf("EPSM search: %f seconds for %d iterations\n", epsm_time, iterations);
        printf("Naive search: %f seconds for %d iterations\n", naive_time, iterations);
        double improvement = (naive_time - epsm_time) / naive_time * 100;
        printf("Performance Improvement: %.2f%%\n", improvement);
        printf("-------------------------------------------------\n");
        
        free(patternBuf);
    }
    
    free(text);
    return 0;
}
