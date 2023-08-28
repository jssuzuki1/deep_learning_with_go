// Unit test importing data into the frame.
package main

import (
	"testing"

	"github.com/petar/GoMNIST"
)

func TestImportData(t *testing.T) {
	// Loads data into [][]float64
	_, _, err := GoMNIST.Load("..\\data")
	if err != nil {
		t.Fatal(err)
	}

}
