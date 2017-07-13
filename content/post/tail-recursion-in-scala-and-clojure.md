---
title: "Tail Recursion In Scala And Clojure"
date: 2016-09-09
tags: []
draft: false
---
I recently read a [blog post](http://www.marcoyuen.com/articles/2016/09/08/stackless-scala-1-the-problem.html "Stackless Scala, Part 1: The Problem") about issues with writing purely functional code in Scala. The post talked a bit about a [paper](http://blog.higher-order.com/assets/trampolines.pdf "Stackless Scala With Free Monads") addressing tail-recursion in Scala. The code example (taken from the paper) was an implementation of zip lists, called zipIndex, that was not properly tail-recursive and would result in a stack overflow for relatively small inputs. A later post from the author will look at ways of addressing the problem and I’m looking forward to reading about it.

I’m wondering if the next post will do something similar to the tail-recursive factorial function.

<pre><code class="language-scala">
def factorial(n: BigInt): BigInt = {
    def fact(n: BigInt, result: BigInt): BigInt = {
        if (n == 0) return result
        else return fact(n - 1, result * n)
    }
    return fact(n, 1)
}
</code></pre>

I’d write a zip list in much the same way:

<pre><code class="language-scala">
def loopRecur[A](index: Int, coll: Seq[A], zippedList: List[(Int, A)]): List[(Int, A)] = {
    if (coll.isEmpty) return zippedList
    else return loopRecur(index + 1, coll.tail, zippedList ++ List((index, coll.head)))
}
// Given a sequence of items, returns a List of tuples of the form (item index, item)
def zipIndex[A](coll: Seq[A]): List[(Int, A)] = {
    return loopRecur(0, coll, List.empty[(Int, A)])
}
</code></pre>

The recursion is handled with the function ```loopRecur```, which is named after Clojure’s [loop](https://clojuredocs.org/clojure.core/loop "loop") and [recur](https://clojuredocs.org/clojure.core/recur "recur") forms. I’ve tested the above implementation of ```zipIndex``` with inputs of up to 100,000 elements. I find the Clojure analog to be somewhat simpler. It's also much faster than the Scala.

<pre><code class="language-clojure">
(defn zipIndex [coll]
  (loop [list coll
         n 0
         result []]
    (if (empty? list)
      result
      (recur (rest list) (inc n) (conj result [n (first list)])))))
</code></pre>

To test the Clojure, assuming you have access to a Clojure REPL, you can run ```(zipList (range 100000))``` and then be amazed at how much faster the Clojure version runs compared to the Scala (assuming the Scala I wrote above is efficient).
