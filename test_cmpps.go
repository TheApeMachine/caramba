package main

import "fmt"

func testCMPPS(a, b float32) uint32

func main() {
	fmt.Printf("CMPPS(1.0, 0.0) = %x\n", testCMPPS(1.0, 0.0))
	fmt.Printf("CMPPS(0.0, 1.0) = %x\n", testCMPPS(0.0, 1.0))
}
