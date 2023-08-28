package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"

	"github.com/e-XpertSolutions/go-iforest/iforest"
	"github.com/petar/GoMNIST"
	"gorgonia.org/gorgonia"
	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// /////////////////
// DEEP LEARNING SECTION
// /////////////////
type nn struct {
	g              *ExprGraph
	w0, w1, wInput *Node

	pred    *Node
	predVal Value
}

func newNN(g *gorgonia.ExprGraph, hiddenSize int) *nn {
	// Create weight matrix for the first layer
	inputSize := 784
	wInputB := tensor.Random(tensor.Float64, inputSize*hiddenSize)
	wInputT := tensor.New(tensor.WithBacking(wInputB), tensor.WithShape(inputSize, hiddenSize))
	wInput := gorgonia.NewMatrix(g,
		tensor.Float64,
		gorgonia.WithName("wInput"),
		gorgonia.WithShape(inputSize, hiddenSize),
		gorgonia.WithValue(wInputT),
	)

	return &nn{
		g:  g,
		w0: wInput,
	}
}

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

	//Loads data into [][]float64
	train, test, err := GoMNIST.Load("..\\data")
	if err != nil {
		log.Fatal(err)
	}

	// Initialize dataframes for images and labels.
	train_images := make([][]float64, len(train.Images))
	train_labels := make([]int, len(train.Images))

	for i := 0; i < len(train.Images); i++ {
		train_images[i] = make([]float64, len(train.Images[0]))
		for p := range train.Images[0] {
			//Integer pixel values 0-255 to float64 0.00-255.00
			train_images[i][p] = float64(train.Images[i][p])
			train_labels[i] = int(train.Labels[i])

		}
	}

	test_images := make([][]float64, len(test.Images))
	test_labels := make([]int, len(test.Images))

	for i := 0; i < len(test.Images); i++ {
		test_images[i] = make([]float64, len(test.Images[0]))
		for p := range test.Images[0] {
			//Integer pixel values 0-255 to float64 0.00-255.00
			test_images[i][p] = float64(test.Images[i][p])
			test_labels[i] = int(test.Labels[i])

		}
	}

	// fmt.Println("No. of Training Images", len(train_images))
	// fmt.Println("No. of Training Labels", len(train_labels))
	// fmt.Println("No. of Test Images", len(test_images))
	// fmt.Println("No. of Test Labels", len(test_labels))

	// Deep learning model parameters

	hiddenSize := 128

	g := NewGraph()
	m := newNN(g, hiddenSize)

	fmt.Print(m)

	xB := make([]float64, len(train_images)*784)
	for i, image := range train_images {
		copy(xB[i*784:], image)
	}

	xT := tensor.New(tensor.WithBacking(xB), tensor.WithShape(len(train_images), 784))
	fmt.Println(xT)

	// Create matrices for 28x28 images.
	x := NewMatrix(g,
		tensor.Float64,
		WithName("X"),
		WithShape(784, 1),
		WithValue(xT),
	)

	fmt.Println(x)

	// Target Values
	yB := train_labels
	yT := tensor.New(tensor.WithBacking(yB), tensor.WithShape(len(train_labels), 1))

	y := NewMatrix(g,
		tensor.Int,
		WithName("y"),
		WithShape(len(train_labels), 1),
		WithValue(yT),
	)

	fmt.Println(y)

	// This part fails due to incompatabilities with the shape. I haven't been able to figure out how to resolve this.

	// Run forward pass
	// // if err := m.fwd(x); err != nil {
	// // 	log.Fatalf("%+v", err)
	// // }

	// I intended to adapt the code below once I got the forward pass to work.

	// // Calculate Cost w/MSE
	// losses := Must(Sub(y, m.pred))
	// square := Must(Square(losses))
	// cost := Must(Mean(square))

	// // Do Gradient updates
	// if _, err = Grad(cost, m.learnables()...); err != nil {
	// 	log.Fatal(err)
	// }

	// // Instantiate VM and Solver
	// vm := NewTapeMachine(g, BindDualValues(m.learnables()...))
	// solver := NewVanillaSolver(WithLearnRate(1.0))

	// for i := 0; i < 10000; i++ {
	// 	vm.Reset()
	// 	if err = vm.RunAll(); err != nil {
	// 		log.Fatalf("Failed at inter  %d: %v", i, err)
	// 	}
	// 	solver.Step(NodesToValueGrads(m.learnables()))
	// 	vm.Reset()
	// }

	///////////////////
	// ISOLATION FOREST
	///////////////////

	// input parameters
	treesNumber := 100
	subsampleSize := 256
	outliersRatio := 0.0001

	// model initialization
	forest := iforest.NewForest(treesNumber, subsampleSize, outliersRatio)

	// Train on the test images data set
	forest.Train(test_images)

	// Test function is necessary to generate Anomaly Scores for Each Sample
	forest.Test(test_images)

	// format of anomalyScores is map[int]float64
	anomalyScores := forest.AnomalyScores

	// Create a dataframe called "AnomalyScores" that has the length of all of the AnomalyScores + 1
	// The +1 is for the header.
	var scores = make([][]string, len(anomalyScores)+1)

	// This for loop goes through every record of "scores" and populates it.
	for i := 0; i < len(scores); i++ {
		scores[i] = make([]string, 2)
		// The first row is the header
		if i == 0 {
			scores[0][0] = "RowID"
			scores[0][1] = "Scores"
		}

		// The second row onward is populated with anomaly scores.
		// Subtract 0.5 since the anomaly scores from iforest are normalized around 0.5 as opposed to 0, allowing for negative values.
		// Because i == 0 is reserved for the header, when i = 1, we want to select the first row of the AnomalyScores, which is the 0th element.
		if i != 0 {
			//Anomaly Scores
			// Multiply by -1
			score := (anomalyScores[i-1] - 0.5) * -1
			scores[i][0] = fmt.Sprintf("%d", i-1)
			scores[i][1] = fmt.Sprintf("%f", score)
		}

	}

	// // Create a new slice to store rows with negative scores
	// var negativeScores [][]string

	// // Iterate through the scores and add rows with negative scores to the new slice
	// for _, row := range scores {
	// 	score, err := strconv.ParseFloat(row[1], 64)
	// 	if err != nil {
	// 		// Handle the error
	// 		continue
	// 	}

	// 	if score < 0 {
	// 		negativeScores = append(negativeScores, row)
	// 	}
	// }

	// Export only the anomalies (negative values)
	file, _ := os.Create("../results/go_scores.csv")
	w := csv.NewWriter(file)
	w.WriteAll(scores)

	//////////////////////////////////////
	// LEFTOVER CODE FROM PREVIOUS ATTEMPT
	//////////////////////////////////////

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
