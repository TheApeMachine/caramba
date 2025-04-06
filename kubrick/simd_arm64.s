#include "textflag.h"

// compareBuffersNEON compares two rune buffers using NEON instructions
TEXT ·compareBuffersNEON(SB), NOSPLIT|NOFRAME, $0-48
    MOVD a+0(FP), R0     // Load address of first buffer
    MOVD b+24(FP), R1    // Load address of second buffer
    MOVD len+16(FP), R2  // Load length of buffers

    // Initialize result to true (1)
    MOVD $1, R3
    MOVD R3, ret+40(FP)

    // Check if length is 0
    CBZ R2, done

    // Main comparison loop
compare_loop:
    MOVD.P 8(R0), R4    // Load 2 runes from first buffer
    MOVD.P 8(R1), R5    // Load 2 runes from second buffer
    CMP R4, R5
    BNE not_equal

    SUB $2, R2, R2      // Decrement counter
    CBNZ R2, compare_loop

done:
    RET

not_equal:
    // Set result to false (0)
    MOVD $0, R3
    MOVD R3, ret+40(FP)
    RET

// clearBufferNEON clears a buffer by setting all values to a specified rune
TEXT ·clearBufferNEON(SB), NOSPLIT|NOFRAME, $0-32
    MOVD dst+0(FP), R0   // Load destination buffer address
    MOVD r+16(FP), R1    // Load rune value to set
    MOVD len+24(FP), R2  // Load length of buffer

    // Check if length is 0
    CBZ R2, clear_done

    // Create a double-rune value for faster clearing
    LSL $32, R1, R3     // Shift first rune to upper 32 bits
    ORR R1, R3          // Combine two runes

clear_loop:
    MOVD.P R3, 8(R0)    // Store 2 runes
    SUB $2, R2, R2      // Decrement counter
    CBNZ R2, clear_loop

clear_done:
    RET

// copyBufferNEON copies runes from source to destination buffer
TEXT ·copyBufferNEON(SB), NOSPLIT|NOFRAME, $0-40
    MOVD dst_base+0(FP), R0   // Load dst base pointer
    MOVD dst_len+8(FP), R1    // Load dst length
    MOVD src_base+24(FP), R2  // Load src base pointer
    MOVD src_len+32(FP), R3   // Load src length

    // Check if src length is greater than dst length
    CMP R3, R1
    BGT panic                 // If src is longer than dst, panic

    // Use shorter length
    MOVD R3, R1

    // Check if length is 0
    CBZ R1, copy_done

    // Copy 8 runes at a time using NEON
    LSR $3, R1, R4          // R4 = length / 8
    CBZ R4, copy_remainder

copy_loop:
    VLD1.P (R2), [V0.D2]    // Load 2 doublewords (4 runes) from src
    VST1.P [V0.D2], (R0)    // Store 2 doublewords to dst
    SUBS $1, R4             // Decrement counter
    BNE copy_loop

copy_remainder:
    // Handle remaining runes (less than 8)
    AND $7, R3, R5         // R5 = remaining runes
    CBZ R5, copy_done

remainder_loop:
    MOVD.P 4(R2), R6      // Load one rune from src
    MOVD.P R6, 4(R0)      // Store one rune to dst
    SUBS $1, R5           // Decrement counter
    BNE remainder_loop

copy_done:
    RET

panic:
    // Just return for now, the Go code will handle the panic
    RET

// findPatternNEON searches for a pattern in a buffer
TEXT ·findPatternNEON(SB), NOSPLIT|NOFRAME, $0-56
    MOVD haystack+0(FP), R0    // Load haystack buffer address
    MOVD needle+24(FP), R1     // Load needle buffer address
    MOVD hlen+16(FP), R2       // Load haystack length
    MOVD nlen+40(FP), R3       // Load needle length

    // Initialize result to -1 (not found)
    MOVD $-1, R4
    MOVD R4, ret+48(FP)

    // Check if needle is longer than haystack
    CMP R2, R3
    BGT pattern_not_found

    // Main search loop
pattern_loop:
    MOVD.P 8(R0), R4    // Load 2 runes from haystack
    MOVD (R1), R5       // Load first rune of needle
    CMP R4, R5
    BEQ check_full_match

    SUB $2, R2, R2      // Decrement counter
    CBNZ R2, pattern_loop

pattern_not_found:
    MOVD $-1, R0
    MOVD R0, ret+48(FP)
    RET

check_full_match:
    // TODO: Implement full pattern matching
    // For now, just return not found
    B pattern_not_found

// findDifferencesNEON finds differences between two rune buffers
TEXT ·findDifferencesNEON(SB), NOSPLIT|NOFRAME, $0-48
    MOVD a+0(FP), R0     // Load first buffer address
    MOVD b+24(FP), R1    // Load second buffer address
    MOVD len+16(FP), R2  // Load length of buffers

    // Initialize result to -1 (no differences)
    MOVD $-1, R3
    MOVD R3, ret+40(FP)

    // Check if length is 0
    CBZ R2, diff_done

diff_loop:
    MOVD.P 8(R0), R4    // Load 2 runes from first buffer
    MOVD.P 8(R1), R5    // Load 2 runes from second buffer
    CMP R4, R5
    BNE diff_found

    SUB $2, R2, R2      // Decrement counter
    CBNZ R2, diff_loop

diff_done:
    RET

diff_found:
    MOVD $0, R3
    MOVD R3, ret+40(FP)
    RET

// findFirstDifferenceNEON finds the first difference between two buffers
TEXT ·findFirstDifferenceNEON(SB), NOSPLIT|NOFRAME, $0-56
    MOVD old+0(FP), R0    // Load old buffer address
    MOVD new+24(FP), R1   // Load new buffer address
    MOVD len+16(FP), R2   // Load buffer length

    // Initialize position counter
    MOVD $0, R3

    // Check if length is 0
    CBZ R2, no_diff

first_diff_loop:
    // Load runes from both buffers
    MOVD (R0), R4
    MOVD (R1), R5
    CMP R4, R5
    BNE found_diff

    // Move to next rune
    ADD $4, R0
    ADD $4, R1
    ADD $1, R3
    SUB $1, R2
    CBNZ R2, first_diff_loop

no_diff:
    // No difference found
    MOVD $-1, R3
    MOVD R3, ret+48(FP)
    RET

found_diff:
    // Return position of first difference
    MOVD R3, ret+48(FP)
    RET

// countRunsNEON counts consecutive identical runes
TEXT ·countRunsNEON(SB), NOSPLIT|NOFRAME, $0-40
    MOVD buf_base+0(FP), R0   // Load buffer address
    MOVD buf_len+8(FP), R1    // Load buffer length
    MOVD ret_base+24(FP), R2  // Load result slice base
    
    // Initialize counters
    MOVD $0, R6              // Result slice index
    
    // Check if buffer is empty
    CBZ R1, done
    
    // Load first rune
    MOVD (R0), R7           // Current value
    MOVD $1, R5             // Start first run
    
    // Decrement length and advance pointer
    SUB $1, R1
    ADD $4, R0
    
count_loop:
    CBZ R1, store_last_run
    
    // Load next rune
    MOVD (R0), R8
    CMP R8, R7
    BEQ continue_run
    
    // Store current run and start new one
    LSL $2, R6, R9          // Multiply index by 4 (size of int32)
    ADD R2, R9, R9          // Add base address
    MOVW R5, (R9)           // Store run length as int32
    ADD $1, R6              // Advance result index
    
    // Start new run
    MOVD R8, R7             // Update current value
    MOVD $1, R5             // Reset run length
    
    B next_rune
    
continue_run:
    ADD $1, R5              // Increment run length
    
next_rune:
    ADD $4, R0              // Advance buffer pointer
    SUB $1, R1              // Decrement counter
    B count_loop
    
store_last_run:
    // Store the last run
    LSL $2, R6, R9          // Multiply index by 4 (size of int32)
    ADD R2, R9, R9          // Add base address
    MOVW R5, (R9)           // Store run length as int32
    ADD $1, R6              // Advance result index
    
done:
    // Return number of runs
    MOVW R6, ret+32(FP)
    RET

// expandRunsNEON expands run-length encoded data
TEXT ·expandRunsNEON(SB), NOSPLIT|NOFRAME, $0-64
    MOVD dst_base+0(FP), R0    // Load destination slice base
    MOVD dst_len+8(FP), R1     // Load destination slice length
    MOVD values_base+24(FP), R2 // Load values slice base
    MOVD values_len+32(FP), R3  // Load values slice length
    MOVD counts_base+48(FP), R4 // Load counts slice base
    
    // Initialize counters
    MOVD $0, R5                // Source index (i)
    MOVD $0, R6                // Destination index (pos)
    
    // Check if input is empty
    CBZ R3, done
    
    // First pass: calculate total length
    MOVD R5, R14               // Save original source index
    MOVD $0, R13               // Total length
    
validate_loop:
    CMP R5, R3
    BGE validate_done
    
    // Load current count
    LSL $2, R5, R7            // R7 = i * 4
    ADD R4, R7, R8            // R8 = counts_base + (i * 4)
    MOVW (R8), R10            // R10 = current count (32-bit)
    SXTW R10, R10             // Sign extend to 64-bit
    
    // Skip if count <= 0
    CMP $0, R10
    BLE next_validate
    
    // Add to total length
    ADD R10, R13              // Add current count to total
    CMP R13, R1              // Compare with destination length
    BGT total_too_large
    
next_validate:
    ADD $1, R5
    B validate_loop
    
validate_done:
    // Reset source index for expansion
    MOVD R14, R5
    
expand_loop:
    // Check if we've processed all values
    CMP R5, R3
    BGE done
    
    // Load current value
    LSL $2, R5, R7            // R7 = i * 4
    ADD R2, R7, R8            // R8 = values_base + (i * 4)
    MOVW (R8), R9             // R9 = current value (32-bit)
    
    // Load current count
    ADD R4, R7, R8            // R8 = counts_base + (i * 4)
    MOVW (R8), R10            // R10 = current count (32-bit)
    SXTW R10, R10             // Sign extend to 64-bit
    
    // Skip if count <= 0
    CMP $0, R10
    BLE next_run
    
    // Store value count times
    MOVD R10, R12            // Copy count to R12 for the inner loop
    
fill_loop:
    CBZ R12, next_run        // If count is 0, move to next run
    
    // Calculate destination address and store value
    LSL $2, R6, R8           // R8 = pos * 4
    ADD R0, R8, R8           // R8 = dst_base + (pos * 4)
    MOVW R9, (R8)            // Store current value (32-bit)
    
    // Update counters
    ADD $1, R6               // Increment destination index
    SUB $1, R12              // Decrement remaining count
    B fill_loop
    
next_run:
    ADD $1, R5               // Move to next value
    B expand_loop
    
total_too_large:
    // Return -1 to indicate error
    MOVD $-1, R6
    B done
    
done:
    // Return total expanded length
    MOVD R6, ret+56(FP)
    RET
