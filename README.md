# Python-EA-Solution-for-TSP
An example of a Evolutionary Algorithm designed to provide solutions for the [Travelling Salesperson Problem](Travelling%20salesman%20problem) .

This is an [NP-Hard](https://en.wikipedia.org/wiki/NP-hardness) problem that is well suited to Evolutionary Algorithms. EA's are able to provide good (though not guaranteed to be optimal) solutions to TSPs with minimal computational expenditure and the TSP is a good representation of an industry-relevant application of Evolutionary Algorithms.

This EA uses tournament selection, and a standard crossover and mutation operator to provide new child solutions which are utilised to replace up to 100% of the gene pool for each new iteration. 

There is also a dedicated and fully customisable Parameters class that the algorithm draws from for testing purposes. 

Matplotlib is used to plot and visualise algorithm results.


