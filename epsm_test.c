#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define ARCH "x86/x64"
    #include <immintrin.h>
    #include <cpuid.h>
#elif defined(__aarch64__) || defined(__arm__)
    #define ARCH "ARM"
    #include <arm_neon.h>
#else
    #define ARCH "Unknown"
#endif

void print_architecture() {
    printf("Architecture: %s\n", ARCH);
}

typedef int (*search_func_t)(const char *text, int n, const char *pattern, int m);

int naive_search(const char *text, int n, const char *pattern, int m) {
    for (int i = 0; i <= n - m; i++) {
        if (memcmp(text + i, pattern, m) == 0)
            return i;
    }
    return -1;
}

//
// ----- x86 SSE4.2 Implementations -----
// These functions are compiled only on x86/x64.
//
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

//
// EPSMa-like variant (epsm_a_opt):
// Uses the first m0 = min(m,8) pattern characters (broadcast into registers)
// and unrolls the inner loop via a switch-case to avoid loop overhead.
//
static inline int epsm_a_opt(const char *text, int n, const char *pattern, int m) {
    int m0 = (m < 8 ? m : 8);
    __m128i B[8];
    for (int i = 0; i < m0; i++) {
        B[i] = _mm_set1_epi8(pattern[i]);
    }
    for (int i = 0; i <= n - m; i++) {
        __m128i block = _mm_loadu_si128((const __m128i*)(text + i));
        switch(m0) {
            case 1:
                if (_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[0])) & 1) {
                    if (memcmp(text + i, pattern, m) == 0) return i;
                }
                break;
            case 2:
                if (((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[0])) & 1) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[1])) & 2) != 0)) {
                    if (memcmp(text + i, pattern, m) == 0) return i;
                }
                break;
            case 3:
                if (((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[0])) & 1) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[1])) & 2) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[2])) & 4) != 0)) {
                    if (memcmp(text + i, pattern, m) == 0) return i;
                }
                break;
            case 4:
                if (((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[0])) & 1) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[1])) & 2) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[2])) & 4) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[3])) & 8) != 0)) {
                    if (memcmp(text + i, pattern, m) == 0) return i;
                }
                break;
            case 5:
                if (((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[0])) & 1) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[1])) & 2) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[2])) & 4) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[3])) & 8) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[4])) & 16) != 0)) {
                    if (memcmp(text + i, pattern, m) == 0) return i;
                }
                break;
            case 6:
                if (((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[0])) & 1) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[1])) & 2) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[2])) & 4) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[3])) & 8) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[4])) & 16) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[5])) & 32) != 0)) {
                    if (memcmp(text + i, pattern, m) == 0) return i;
                }
                break;
            case 7:
                if (((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[0])) & 1) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[1])) & 2) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[2])) & 4) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[3])) & 8) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[4])) & 16) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[5])) & 32) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[6])) & 64) != 0)) {
                    if (memcmp(text + i, pattern, m) == 0) return i;
                }
                break;
            case 8:
                if (((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[0])) & 1) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[1])) & 2) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[2])) & 4) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[3])) & 8) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[4])) & 16) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[5])) & 32) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[6])) & 64) != 0) &&
                    ((_mm_movemask_epi8(_mm_cmpeq_epi8(block, B[7])) & 128) != 0)) {
                    if (memcmp(text + i, pattern, m) == 0) return i;
                }
                break;
            default:
                break;
        }
    }
    return -1;
}

// EPSMb-like variant (epsm_b_opt): processes text in 16-byte blocks and blends adjacent blocks.
static inline int epsm_b_opt(const char *text, int n, const char *pattern, int m) {
    int m0 = (m < 8 ? m : 8);
    for (int i = 0; i <= n - m; i += 16) {
        __m128i block = _mm_loadu_si128((const __m128i*)(text + i));
        for (int j = 0; j <= 16 - m0; j++) {
            if (i + j > n - m)
                break;
            if (memcmp(text + i + j, pattern, m0) == 0) {
                if (m == m0 || memcmp(text + i + j, pattern, m) == 0)
                    return i + j;
            }
        }
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

// EPSMc-like variant remains as before.
static inline int epsm_c(const char *text, int n, const char *pattern, int m) {
    if (m < 16)
        return naive_search(text, n, pattern, m);
    
    const int alpha = 16;
    const int k = 11;
    int mask = (1 << k) - 1;
    int num_candidates = m - alpha + 1;
    int *candidates = malloc(num_candidates * sizeof(int));
    int *hashes = malloc(num_candidates * sizeof(int));
    if (!candidates || !hashes) {
        free(candidates);
        free(hashes);
        return naive_search(text, n, pattern, m);
    }
    for (int i = 0; i < num_candidates; i++) {
        int h = 0;
        for (int j = 0; j < alpha; j++) {
            h += (unsigned char)pattern[i+j];
        }
        h = h & mask;
        candidates[i] = i;
        hashes[i] = h;
    }
    
    int pos_found = -1;
    for (int i = 0; i <= n - alpha; i += alpha) {
        int h = 0;
        for (int j = 0; j < alpha; j++) {
            h += (unsigned char)text[i+j];
        }
        h = h & mask;
        for (int k_i = 0; k_i < num_candidates; k_i++) {
            if (hashes[k_i] == h) {
                int pos = i - candidates[k_i];
                if (pos >= 0 && pos <= n - m) {
                    if (memcmp(text + pos, pattern, m) == 0) {
                        pos_found = pos;
                        goto cleanup;
                    }
                }
            }
        }
    }
cleanup:
    free(candidates);
    free(hashes);
    return pos_found;
}

// Unified EPSM search for x86: choose variant based on pattern length.
static inline int epsm_search_x86(const char *text, int n, const char *pattern, int m) {
    if (m <= 8)
        return epsm_b_opt(text, n, pattern, m);
    else if (m <= 32)
        return epsm_a_opt(text, n, pattern, m);
    else if (m <= 128)
        return epsm_c(text, n, pattern, m);
    else
        return naive_search(text, n, pattern, m);
}
#define epsm_search epsm_search_x86

//
// ----- End x86 Implementations -----
//
#elif defined(__aarch64__) || defined(__arm__)
// ARM NEON implementation.
static inline int epsm_neon(const char *text, int n, const char *pattern, int m) {
    if (m > 16)
        return naive_search(text, n, pattern, m);
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
static inline int epsm_search(const char *text, int n, const char *pattern, int m) {
    return epsm_neon(text, n, pattern, m);
}
#else
static inline int epsm_search(const char *text, int n, const char *pattern, int m) {
    return naive_search(text, n, pattern, m);
}
#endif

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
// Main: Benchmark across various pattern lengths on a large text.
//
int main() {
    print_architecture();

    int textSize = 1000000;
    char *text = malloc(textSize + 1);
    if (!text) {
        fprintf(stderr, "Failed to allocate text buffer.\n");
        return 1;
    }
    const char *sampleText = "This is a sample text for testing the EPSM algorithm. "
                             "We want to see if the pattern is found using SIMD! ";
    generate_text(text, textSize, sampleText);
    printf("Text size: %d characters\n", textSize);
    printf("-------------------------------------------------\n");

    int patLens[] = {4, 8, 16, 32, 64};
    int numPatLens = sizeof(patLens) / sizeof(patLens[0]);
    for (int p = 0; p < numPatLens; p++) {
        int m = patLens[p];
        if (m > textSize) break;
        const char *pattern = text + 50;  // Choose a pattern from a known offset.
        char *patternBuf = malloc(m + 1);
        if (!patternBuf) continue;
        memcpy(patternBuf, pattern, m);
        patternBuf[m] = '\0';

        printf("Pattern (length %d): \"%s\"\n", m, patternBuf);

        int pos_epsm = epsm_search(text, textSize, patternBuf, m);
        int pos_naive = naive_search(text, textSize, patternBuf, m);
        if (pos_epsm >= 0)
            printf("EPSM: Pattern found at position %d\n", pos_epsm);
        else
            printf("EPSM: Pattern not found.\n");
        if (pos_naive >= 0)
            printf("Naive: Pattern found at position %d\n", pos_naive);
        else
            printf("Naive: Pattern not found.\n");

        int iterations = (m <= 8 ? 100000 : (m <= 32 ? 10000 : 1000));
        clock_t start, end;
        double epsm_time, naive_time;

        start = clock();
        for (int i = 0; i < iterations; i++) {
            volatile int dummy = epsm_search(text, textSize, patternBuf, m);
            (void)dummy;
        }
        end = clock();
        epsm_time = (double)(end - start) / CLOCKS_PER_SEC;

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
