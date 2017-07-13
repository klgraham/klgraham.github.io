---
title: "Probability Monad in Swift"
date: 2016-03-25
tags: []
draft: false
---
I read about the probability monad in [The Frequentist Approach to Probability](http://jliszka.github.io/2013/08/12/a-frequentist-approach-to-probability.html) a couple of years ago and thought it was pretty neat. I decided to make one in Swift, as an exercise in learning the language, after having done the same in [Clojure](https://github.com/klgraham/watershed) [and Java 8](https://gist.github.com/klgraham/c1bc8fb6accb97e5aa6f "Probability Monad in Java 8").

If you don’t know what a monad is, don’t worry, it’s not important for this post. Just know that it lets you do neat stuff with probability distributions, programmatically speaking. Ok… on to the code.

## Protocols for Probabilities

I initially tried to do this with two protocols. I was feeling very gung-ho about protocol-oriented programming and was thinking of Swift's protocols like Scala's traits. I had one protocol that described randomly sampling values from a probability distribution

<pre><code>
protocol Stochastic {
  // the type of value stored in the distribution
  associatedtype ValueType

  // Sample a single value from the distribution
  func get() -> ValueType

  // Sample n values from the distribution
  func sample(n: Int) -> [ValueType]
}
</code></pre>

and another one that allowed for parameterization, for things like the Poisson distribution.

<pre><code>
protocol Parameterized {
    associatedtype ParameterType
    var p: ParameterType { get }
}
</code></pre>

You may have noticed that there’s no protocol that allows you to map one distribution into another, which is what would make this into a monad. That’s because I had not yet figured out how to do it with structs or classes. It’s easy to map one set of values drawn from a distribution into another set of values according to a function. But I really needed to create a new struct with a specific `get()` function. And then I remembered that functions were first class values in Swift!

Turns out you don’t need protocols or classes for this at all. You can do it all with a pretty simple struct!

## Probability Distributions via Closures

With a single generic struct, we have everything we need for the probability monad. To convert one distribution into another, we need only pass in a function that maps elements of one distribution into elements of the other.

<pre><code>
struct Distribution&lt;A&gt; {
    var get: () -> A?

    func sample(n: Int) -> [A] {
        return (1...n).map { x in get()! }
    }

    func map&lt;B&gt;(using f: @escaping (A) -> B) -> Distribution&lt;B&gt; {
        var d = Distribution&lt;B&gt;(get: {() -> Optional&lt;B&gt; in return nil})
        d.get = {
            (Void) -> B in return f(self.get()!)
        }
        return d
    }
}
</code></pre>

First, to do any sampling, we need to generate random numbers. Swift offers `arc4random` for this. If you're on macOS, you can use the Foundation framework (unsure if `arc4random` is on Linux). You could also use [GameplayKit’s GKRandomSource](https://www.hackingwithswift.com/read/35/3/generating-random-numbers-with-gameplaykit-gkrandomsource) on any of the Apple platforms. Or you can always use something from C.

<pre><code>
func randomDouble() -> Double {
  return Double(arc4random()) / Double(UInt32.max)
}
</code></pre>

Let’s see what we can do if we start from the [Uniform distribution](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)). Below, we pass the random number generator function to the distribution's constructor.

<pre><code>
let uniform = Distribution&lt;Double&gt;(get: randomDouble)
uniform.sample(5)
</code></pre>

For starters, we can easily generate the true-false distribution by mapping the Uniform distribution with a function that generates boolean values from a double. From there, it’s straightforward to transform the true-false distribution into the Bernoulli distribution

<pre><code>
let tf = uniform.map({ $0 < 0.7 })
tf.sample(10)

let bernoulli = tf.map({ $0 ? 1 : 0 })
bernoulli.sample(10)
</code></pre>

By passing the appropriate transformation function or closure to the `map` function, one type of distribution can be converted into another.

## Composing Distributions

If you want to compose distributions, then our Distribution struct needs to use a `flatMap` function to map one distribution into another. This works by composing the input function `f` so that when values are sampled, the values are directly drawn from the new distribution.

<pre><code>
func map&lt;B&gt;(using f: @escaping (A) -> Distribution&lt;B&gt;) -> Distribution&lt;B&gt; {
    var d = Distribution&lt;B&gt;(get: { () -> Optional&lt;B&gt; in return nil })
    d.get = {
        (Void) -> B in return f(self.get()!).get()!
    }
    return d
}
</code></pre>

One example of composing distributions comes from combining a pair of six-sided dice. We can start with a single die. (For this, we need to generate random integers)

<pre><code>
func nextInt(min min: Int, max: Int) -> ((Void) -> Int) {
     assert(max > min)
     return { () in return Int(arc4random_uniform(UInt32((max - min) + 1))) + min }
 }

 let die6 = Distribution&lt;Int&gt;(get: nextInt(min: 1, max: 6))
 die6.sample(10)
 </code></pre>

Next, we can use the flatMap function to compose the distributions of a pair of six-sided dice by passing in a function that combines the rolls of a pair of dice.

<pre><code>
let dice = die6.flatMap({
  (d1: Int) in return die6.map({ (d2: Int) in return d1 + d2 })
})

dice.sample(10)
</code></pre>

Now that you’ve seen all the pieces, here’s the final form of the probability distribution struct:

<pre><code>
struct Distribution&lt;A&gt; {
    var get: () -> A?

    func sample(n: Int) -> [A] {
        return (1...n).map { x in get()! }
    }

    func map&lt;B&gt;(using f: @escaping (A) -> B) -> Distribution&lt;B&gt; {
        var d = Distribution&lt;B&gt;(get: { () -> Optional&lt;B&gt; in return nil })
        d.get = {
            (Void) -> B in return f(self.get()!)
        }
        return d
    }

    func mapDistribution&lt;B&gt;(using f: @escaping (A) -> Distribution&lt;B&gt;) -> Distribution&lt;B&gt; {
        var d = Distribution&lt;B&gt;(get: { () -> Optional&lt;B&gt; in return nil })
        d.get = {
            (Void) -> B in return f(self.get()!).get()!
        }
        return d
    }

    let N = 10000
    func prob(of predicate: (A) -> Bool) -> Double {
        return Double(sample(n: N).filter(predicate).count) / Double(N)
    }

    func mean() -> Double {
        return sample(n: N).reduce(0, { $0 + Double(String(describing: $1))! }) / Double(N)
    }

    func variance() -> Double {
        var sum: Double = 0
        var sqrSum: Double = 0

        for x in sample(n: N) {
            let xx = Double(String(describing: x))!
            sum += xx
            sqrSum += xx * xx
        }

        return (sqrSum - sum * sum / Double(N)) / Double(N-1)
    }

    func stdDev() -> Double {
        return sqrt(self.variance())
    }
}
</code></pre>

## Computing Statistics for a Distribution

You may have noticed a few functions for summary statistics (mean, etc) and probability computation. The most important function is `prob`, which lets you use predicates to ask questions of the distribution. There are a few basic examples of what you can do below.

<pre><code>
uniform.mean() // Approximately 0.5

uniform.prob(of: { $0 > 0.7 })   // Approximately 0.3

dice.prob(of: { $0 == 7 })   // Approximately 1/6
</code></pre>

If I can figure how to properly implement the `given()` function I’ll add that in a future post. I also want to be able to handle more interesting distributions, like that seen in a probabilistic graphical model.

The code is at https://github.com/klgraham/probability-monad.
