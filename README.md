Martin Krebs, 514407

# FASHION MNIST FROM SCRATCH

- solved using templates and recursive types
- allows for optimalizations on compilation, everything is laid out in memory
- not using paralellism atm, does not give performance benefits for task
- the network is trained as fully connected MLP, with SGD using momentum and decay

### NOTES FROM WORKING ON THE PROJECT

- if it does not work on XOR, it will not work on mnist holds
- std::array is on stack
- not violating 1st rule helps, [thanks Andrej](https://twitter.com/karpathy/status/1013244313327681536)
- backprop is quite easy to comprehend, not that easy to implement

### NEXT STEPS
- i think i learned quite a lot during the implementation process, i tried to implement CNN as well, but I failed miserably
- i would like to finish implementing it
- same with dropout, and recurrent network
- add proper paralellism, may work for larger networks
- the whole solution lacks quite a modularity
  - in the beginning, I was inspired by torch modules, and I had separate module for Linear layer, ReLU and Softmax
  - turned out that the network was much slower (because of the double number of layers), so i ditched the original idea
  - i would like to make it somehow work, but not sure how to avoid the slow down
