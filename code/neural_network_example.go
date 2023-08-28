package main

import (
	"fmt"
	"log"
	"math/rand"

	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var err error

type nn struct {
	g      *ExprGraph
	w0, w1 *Node

	pred    *Node
	predVal Value
}

// // Takes gorgonia graph
// func newNN(g *ExprGraph) *nn {
// 	// Create node for w/weight
// 	wB := tensor.Random(tensor.Float64, 3)
// 	wT := tensor.New(tensor.WithBacking(wB), tensor.WithShape(3, 1))

// 	w0 := NewMatrix(g,
// 		tensor.Float64,
// 		WithName("w"),
// 		WithShape(3, 1),
// 		WithValue(wT),
// 	)
// 	return &nn{
// 		g:  g,
// 		w0: w0,
// 	}
// }

// func newNN(g *ExprGraph, inputShape tensor.Shape) *nn {
// 	// Create weight matrix for the first layer

// 	hiddenSize := 10
// 	w0B := tensor.Random(tensor.Float64, inputShape[1]*hiddenSize) // Adjust hiddenSize as needed
// 	w0T := tensor.New(tensor.WithBacking(w0B), tensor.WithShape(inputShape[1], hiddenSize))
// 	w0 := NewMatrix(g,
// 		tensor.Float64,
// 		WithName("w0"),
// 		WithShape(inputShape[1], hiddenSize),
// 		WithValue(w0T),
// 	)

// 	return &nn{
// 		g:  g,
// 		w0: w0,
// 	}
// }

func newNN(g *ExprGraph, inputShape tensor.Shape) *nn {
	hiddenSize := 10
	w0B := tensor.Random(tensor.Float64, inputShape.TotalSize()*hiddenSize)
	w0T := tensor.New(tensor.WithBacking(w0B), tensor.WithShape(inputShape.TotalSize(), hiddenSize))
	w0 := NewMatrix(g,
		tensor.Float64,
		WithName("w0"),
		WithShape(inputShape.TotalSize(), hiddenSize),
		WithValue(w0T),
	)

	return &nn{
		g:  g,
		w0: w0,
	}
}

// func newNN(g *ExprGraph, inputShape tensor.Shape, hiddenSize int) *nn {
// 	w0B := tensor.Random(tensor.Float64, inputShape.TotalSize()*hiddenSize)
// 	w0T := tensor.New(tensor.WithBacking(w0B), tensor.WithShape(inputShape.TotalSize(), hiddenSize))

// 	w0 := NewMatrix(g,
// 		tensor.Float64,
// 		WithName("w0"),
// 		WithShape(inputShape.TotalSize(), hiddenSize),
// 		WithValue(w0T),
// 	)

// 	return &nn{
// 		g:  g,
// 		w0: w0,
// 	}
// }

func (m *nn) learnables() Nodes {
	return Nodes{m.w0}
}

func (m *nn) fwd(x *Node) (err error) {
	var l0, l1 *Node

	// Set first layer to be copy of input
	l0 = x

	// Dot product of l0 and w0, use as input for Sigmoid
	l0dot := Must(Mul(l0, m.w0))

	// Build hidden layer out of result
	l1 = Must(Sigmoid(l0dot))

	m.pred = l1
	Read(m.pred, &m.predVal)
	return nil

}

func main() {

	rand.Seed(31337)

	// Create a flattened array xB representing two (4, 3) matrices
	xB := []float64{
		0, 0, 1,
		0, 1, 1,
		1, 0, 1,
		1, 1, 1,
		0, 0, 1,
		0, 1, 0,
		1, 0, 0,
		1, 1, 0,
	}
	// Reshape xB to match (2, 4, 3) shape
	xT := tensor.New(tensor.WithBacking(xB), tensor.WithShape(2, 4, 3))

	g := NewGraph()
	inputShape := tensor.Shape{2, 4, 3} // Set the desired input shape
	m := newNN(g, inputShape.Dims())
	// m := newNN(g, inputShape)

	x := NewMatrix(g,
		tensor.Float64,
		WithName("X"),
		WithShape(2, 4, 3),
		WithValue(xT),
	)

	// Set input x to network
	// xB := []float64{0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1}
	// xT := tensor.New(tensor.WithBacking(xB), tensor.WithShape(4, 3))
	// x := NewMatrix(g,
	// 	tensor.Float64,
	// 	WithName("X"),
	// 	WithShape(4, 3),
	// 	WithValue(xT),
	// )

	// Create a batch of input data (matrix X)
	// This works.

	// Define validation data set
	yB := []float64{0, 0, 1, 1}
	yT := tensor.New(tensor.WithBacking(yB), tensor.WithShape(4, 1))
	y := NewMatrix(g,
		tensor.Float64,
		WithName("y"),
		WithShape(4, 1),
		WithValue(yT),
	)

	// Run forward pass
	if err := m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	// Calculate Cost w/MSE
	losses := Must(Sub(y, m.pred))
	square := Must(Square(losses))
	cost := Must(Mean(square))

	// Do Gradient updates
	if _, err = Grad(cost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	// Instantiate VM and Solver
	vm := NewTapeMachine(g, BindDualValues(m.learnables()...))
	solver := NewVanillaSolver(WithLearnRate(1.0))

	for i := 0; i < 10000; i++ {
		vm.Reset()
		if err = vm.RunAll(); err != nil {
			log.Fatalf("Failed at inter  %d: %v", i, err)
		}
		solver.Step(NodesToValueGrads(m.learnables()))
		vm.Reset()
	}
	fmt.Println("\n\nOutput after Training: \n", m.predVal)
}


// LEFTOVER CODE FROM PREVIOUS PROGRAM


	//
	//
	//
	////
	//
	////
	//
	//
	//
	//
	////
	//
	////
	//	//
	//
	//
	////
	//
	////
	//	//
	//
	//
	////
	//
	////
	//
	// OLD ATTEMPT

	// // Define the neural network architecture
	// inputSize := len(train_images[0])
	// hiddenSize := 128
	// outputSize := 10 // Assuming 10 classes for MNIST dataset

	// // Input data shape parameters
	// x := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, inputSize), gorgonia.WithName("x"), gorgonia.WithInit(gorgonia.GlorotU(1)))

	// // Weight matrces
	// w1 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(inputSize, hiddenSize), gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotU(1)))
	// w2 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(hiddenSize, outputSize), gorgonia.WithName("w2"), gorgonia.WithInit(gorgonia.GlorotU(1)))

	// // bias values
	// b1 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, hiddenSize), gorgonia.WithName("b1"), gorgonia.WithInit(gorgonia.Zeroes()))
	// b2 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(1, outputSize), gorgonia.WithName("b2"), gorgonia.WithInit(gorgonia.Zeroes()))

	// fmt.Println(x)
	// fmt.Println(w1)
	// fmt.Println(w2)
	// fmt.Println(b1)
	// fmt.Println(b2)

	// // Neural network operations
	// z1 := gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Mul(x, w1)), b1))
	// a1 := gorgonia.Must(gorgonia.Rectify(z1))
	// z2 := gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Mul(a1, w2)), b2))
	// softmax := gorgonia.Must(gorgonia.SoftMax(z2))
	// fmt.Println(softmax)

	// // Calculate one-hot encoded labels for classification.
	// oneHotLabels := make([][]float64, len(train_labels))
	// for i, label := range train_labels {
	// 	oneHot := make([]float64, 10)
	// 	oneHot[label] = 1.0
	// 	oneHotLabels[i] = oneHot
	// }
	// fmt.Println(oneHotLabels)

	// // fmt.Println(oneHotLabels)
	// // fmt.Println(train_labels[59999])

	// // Convert one-hot labels to a Gorgonia tensor
	// oneHotTensor := gorgonia.NewTensor(g, tensor.Float64, 2, gorgonia.WithShape(len(train_labels), 10), tensor.WithBacking(oneHotLabels))
	// fmt.Println(oneHotTensor)

	// // Create a new tensor with the provided one-hot encoded labels
	// oneHotTensorData := make([]float64, len(train_labels)*numClasses)
	// for i, oneHot := range oneHotLabels {
	// 	copy(oneHotTensorData[i*numClasses:], oneHot)
	// }

	// oneHotTensor := gorgonia.NewTensor(g, tensor.Float64, 2, gorgonia.WithShape(len(train_labels), numClasses))
	// oneHotT := oneHotTensor.(*tensor.Dense)
	// oneHotT.MemcpyFromHost(oneHotTensorData) // Copy the data from oneHotTensorData to the tensor

	// Define loss function
	// oneHotLabels := gorgonia.Must(gorgonia.OneHot(g, tensor.Int, gorgonia.Shape{len(train_labels), numClasses}, train_labels...))
	// loss := gorgonia.Must(gorgonia.Mean(gorgonia.Must(gorgonia.Neg(gorgonia.Must(gorgonia.Mean(gorgonia.Must(gorgonia.HadamardProd(gorgonia.Must(gorgonia.Log(softmax)), oneHotLabels))))))))

	// // Define gradient descent optimizer
	// grads, err := gorgonia.Gradient(loss, w1, w2, b1, b2)
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// // Define update operations
	// learningRate := 0.001
	// vm := gorgonia.NewTapeMachine(g)
	// defer vm.Close()

	// for epoch := 0; epoch < 10; epoch++ {
	// 	for i := 0; i < len(train_images); i++ {
	// 		xVal := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(1, inputSize), tensor.WithBacking(train_images[i]))
	// 		_, err := vm.RunAll(x.Bind(xVal))
	// 		if err != nil {
	// 			log.Fatal(err)
	// 		}

	// 		vm.Reset()

	// 		// Update weights using gradients and learning rate
	// 		for _, grad := range grads {
	// 			gradVal, err := grad.Value()
	// 			if err != nil {
	// 				log.Fatal(err)
	// 			}
	// 			gradVal.MulScalar(learningRate)
	// 			grad.Set(gradVal)
	// 			grad.UseUnsafe()
	// 		}
	// 	}

	// 	fmt.Printf("Epoch %d: Loss %v\n", epoch+1, loss.Value())

	// 	// Shuffle data for the next epoch
	// 	vecf64.Shuffle(train_images, func(i, j int) {
	// 		train_images[i], train_images[j] = train_images[j], train_images[i]
	// 		train_labels[i], train_labels[j] = train_labels[j], train_labels[i]
	// 	})
	// }

}
