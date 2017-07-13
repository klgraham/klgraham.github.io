---
title: "Building A Simple Neural Network in Swift"
date: 2016-04-05
tags: []
draft: false
---
So… you want to learn about neural networks? Well, you’ve come to the right place.

This post won’t focus on the theory behind how neural nets work. There are already several excellent websites for that, like Michael Nielsen’s [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html). This focuses on building a neural net in code. So, if you want to skip straight to the code, [the repo is on Github](https://github.com/klgraham/simple-neural-networks).

## What is a Neural Network?

There are many ways to answer this question, but the answer that resonates most deeply with me, and is perhaps most fundamental, is that a neural network is a function approximator. It transforms its input into its output. One of my college professors wrote a paper, [Approximation by superpositions of a sigmoidal function](http://www.dartmouth.edu/~gvc/Cybenko_MCSS.pdf), proving that neural networks can approximate *any* function. This capability is what makes neural networks so powerful and exciting. And all you need to do is select the right weights (it’s not quite that simple).

Note: Links to several papers on the theoretical underpinnings of neural networks are at the bottom of this post.

## The Simplest Neural Network

![](/images/perceptron.png)

We’ll start with a single perceptron, the simplest model of a neuron. Depending on the size of the inputs, the output is either 1 or -1. An output of 1 means the perceptron is on, -1 means the perceptron is off. (Usually, a perceptron’s output is made to be 1 or 0, but 1 or -1 is more convenient for this example.) For our simple example, there’s one input, `x`, which has weight `w`. To determine the output, called the activation, we first take the dot product of the input and weight vectors, then pass the result through the sign function to get the activation.

<pre><code>
func sign(z: Double) -> Int {
    return (z > 0) ? 1 : -1
}
</code></pre>

<pre><code>
func computeActivation(x: [Double], _ w: [Double]) -> Int {
    var sum = 0.0
    for i in 0..&lt;x.count {
        sum += x[i] * w[i]
    }
    return sign(sum)
}
</code></pre>

You may be wondering why we have to use the [sign function](https://en.wikipedia.org/wiki/Sign_function). This is just because we want the output of the perceptron to be 1 or -1. For other problems we might want to have a wider range of output values. In such cases we would replace the sign function with something else, like the [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) or [arctangent](https://en.wikipedia.org/wiki/Inverse_trigonometric_functions). In general though, the activation functions used in neural networks take real-valued input and return output that is limited to specific range or to a set of specific values. There are exceptions to this statement as well.

## A Simple Problem

Let's apply the perceptron to simple binary classification problem. That is, to classify an input as belonging to one of two categories. In such a case, we can map each category to one of the two possible output values of the perceptron. Let’s consider the case where `x` is zero. In such a case it doesn’t matter what the weights are set to, the activation of the perceptron will always be off. That’s not good. To combat such problems we add a fixed input to the perceptron, called the bias. The addition of the bias slightly changes the code for the activation function.

<pre><code>
func computeActivation(bias: Double, bW: Double, x: [Double], w: [Double]) -> Int {
    var sum = 0.0
    for i in 0..&lt;x.count {
        sum += x[i] * w[i]
    }
    sum += bias * bW
    return step(sum)
}
</code></pre>

Now, let’s put these pieces together into a Perceptron class:

<pre><code>
func randomDouble() -> Double {
    return Double(arc4random()) / Double(UINT32_MAX)
}

class Perceptron {
    var bias: Double

    // weights[0] is the weight for the bias input
    var weights: [Double]

    init(numInputs: Int, bias: Double) {
        self.bias = bias

        self.weights = [Double]()
        for _ in 0...numInputs{
            weights.append(1.0 * (randomDouble() - 0.5))
        }
    }

    func activate(input: [Double]) -> Int {
        assert(input.count + 1 == weights.count)

        var sum = 0.0
        for i in 0..&lt;input.count {
            sum += input[i] * weights[i + 1]
        }
        sum += bias * weights[0]
        return (sum > 0) ? 1 : -1
    }
}
</code></pre>

But how do we pick the right weights? The answer is that we don’t. There’s an algorithm for that, backpropagation. Backpropagation is a fancy way of saying that we:

1. See how far away the prediction of our network is from the expected output.
2. Take a step in weight parameter space in the direction that minimizes the error. If you remember your calculus lessons, this is a step in the negative gradient direction. How a big a step to take depends on the size of the error and on how fast we want to move in that direction. We don’t want to take too big a step or too small step. In the former case, we can easily shoot past the optimal weights and in the latter case we might take a long time to get there.  
3. Return to step 1 and repeat until the error is “small enough”.

We could write steps 1 and 2 in code as

<pre><code>
func backProp(input: [Double], output: Int) {
    let prediction = feedForward(input)
    let error = output - prediction

    for i in 0..&lt;weights.count {
        weights[i] += learningRate * Double(error) * input[i]
    }
}
</code></pre>

The full training happens when we pass a sequence of input, output pairs to the backProp function. With each call to backProp the weights of the perceptron are altered to decrease future errors. To handle the training, I’ve made a PerceptronTrainer struct and a struct to hold the training data as well.

<pre><code>
struct PerceptronDataPair {
    let input: [Double]
    let output: Int
}

struct PerceptronTrainer {
    let data: [PerceptronDataPair]

    func train( p: inout Perceptron) -> Int {
        var error: Int = 0
        var count = 0

        for d in data {
            error = p.backProp(input: d.input, output: d.output)
            count += 1
        }

        return error
    }
}
</code></pre>

### Training the Perceptron

We want our perceptron to tell us if a given point is above or below a line in the xy plane. You can pick any line you want to, but I’ll take a simple one like `y(x) = 3x + 1`. We can generate training data by

1. picking N input values at random and computing the y value for each
2. Determine whether the y value is above or below the line.

Then, create a PerceptronTrainer and pass the training data to it and call the train function.

<pre><code>
func createData(numPoints: Int) -> [PerceptronDataPair] {
    var data = [PerceptronDataPair]()

    for _ in 0..&lt;numPoints {
        let x = [2.0 * (randomDouble() - 0.5)]
        let y = line(x[0])
        data.append(PerceptronDataPair(input: x, output: isAbove(y)))
    }

    return data
}

let trainingData = createData(100)

let trainer = PerceptronTrainer(data: data)
trainer.train(&p)
</code></pre>

### How Well Does the Perceptron Work?

Let’s pass 100 random inputs to the perceptron and see how often the predictions are correct. We’ll also create a new, untrained perceptron and see how often it’s predictions are correct.

<pre><code>
let testData = createData(numPoints: 100)

func evaluatePerceptron(p: Perceptron, testData: [PerceptronDataPair]) -> Double {
    var correct = 0
    for d in testData {
        let prediction = p.feedForward(input: d.input)
        if (prediction == d.output) {
            correct += 1
        }
    }

    return Double(correct) / Double(testData.count)
}

// The % correct will be much higher than for an untrained perceptron
evaluatePerceptron(p: p, testData: testData)

let pUntrained = Perceptron(numInputs: 1, offState: -1, bias: 1)
evaluatePerceptron(p: pUntrained, testData: testData)
</code></pre>

I get anywhere from 88-100% correct for the trained perceptron and about 4-40% correct for the untrained perceptron. Not bad for a simple neural network and a simple problem.


There were several papers written around the same time that look into how a neural network is a universal function approximator. They’re behind a paywall, but you can probably get them on Sci-Hub.

- [Approximation by superpositions of a sigmoidal function](http://www.dartmouth.edu/~gvc/Cybenko_MCSS.pdf)
- [Multilayer feedforward networks are universal approximators](http://www.sciencedirect.com/science/article/pii/0893608089900208)
- [Universal approximation of an unknown mapping and its derivatives using multilayer feedforward networks](http://www.sciencedirect.com/science/article/pii/0893608090900056)
- [On the approximate realization of continuous mappings by neural networks](http://www.sciencedirect.com/science/article/pii/0893608089900038)
- [Approximation capabilities of multilayer feedforward networks](http://www.sciencedirect.com/science/article/pii/089360809190009T)
