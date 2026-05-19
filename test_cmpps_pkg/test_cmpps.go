package main

import "fmt"

func testCMPPS(a, b float32) uint32

func main() {
	fmt.Printf("CMPSS $6, X1, X0 (X0=a, X1=b)\n")
	fmt.Printf("a=1.0, b=0.0 -> %x\n", testCMPPS(1.0, 0.0))
	fmt.Printf("a=0.0, b=1.0 -> %x\n", testCMPPS(0.0, 1.0))
}
