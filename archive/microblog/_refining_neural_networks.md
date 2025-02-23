# On refining neural networks 
*Request: This is work in progress and the blogging equivalent of [rubber duck debugging](https://en.wikipedia.org/wiki/Rubber_duck_debugging).  Please please please don't share this forward.*

## Overview

A recent line of work in program synthesis looks at extracting programmatic interpretable representations using a neural network as an admissible heuristic. Their results show that they can derive a compression of the neural representation as a programmatic representation and achieve approximately the same accuracy. An interesting consequence of this compression is that the resultant program -- along with its interpretation -- uses an order of magnitude lesser parameters than the neural network that the program was derived from.

I wanted to pose a naive algorithm to operationalize program synthesis for compressing computer vision models. Specifically, we observe that the building blocks of a convolutional neural network -- the convolutional kernels -- often learn semantically meaningful filters that can be expressed in a differentiable DSL. This project aims to develop a neural search technique to find the best approximation of a trained convolutional network using symbolic primitives defined in this differentiable DSL of CNN kernels.


The original idea stems from a report from a class on Program Synthesis.