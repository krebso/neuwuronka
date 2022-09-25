# neuwuronka

## PV021 project -- Deep Learning from Scratch

### DEADLINE
- Thursday 1. 12. 2022 23:59 (December 1st)

### TASK
- Implement a neural network in a low-level programming language (Rust)
    without the use of advanced libraries or frameworks.
- Train it on the supplied Fashion-MNIST dataset using a backpropagation
   algorithm.

### REQUIREMENTS
  - Your solution must follow the project template (will be provided
    in the study materials) and must be runnable on the Aisa server.
  - Your solution must achieve at least 88% accuracy on the test set.
  - Your solution must finish within 30 minutes. (parse inputs, train, evaluate, export results.)
  - Your solution must contain a runnable script called `run.sh` (not `run`, not
   `RUN.sh`, not `run.exe` etc.) which compiles and executes your code
    (and exports the results).
  - Your solution must output two files to the root project directory:
    (next to `example_test_predictions.csv` file):
     - `train_predictions.csv` - your network predictions for the train set.
     - `test_predictions.csv` - your network predictions for the test set.
  - The format of these files has to be the same as the supplied
    training/testing labels:
     - One prediction per line.
     - Prediction for i-th input vector (ordered by the input .csv file) must
       be on i-th line in the associated output file.
     - Each prediction is a single integer 0 - 9.
  - Replace the first line of this file with UČOs and names
  - Submit your solution in a .zip format before the deadline to a vault in IS.
    We will let you know once we create it.

### SCORING
  - The project has binary scoring -- either you pass it or fail it.
    It is required to pass it in order to get your mark from the oral exam
    and pass the subject.
  - All submitted source files will be checked manually.
  - All submitted solutions will be also checked by an automatic evaluator.
    A simple evaluator is provided in the project template as well.
  - All submitted solutions will be checked for plagiarism.
    Submitting someone else's solution (e.g. publicly available solutions)
    as your own will result in failure and will lead to disciplinary action.
  - Any implementation that uses testing input vectors for anything else
    than final evaluation will result in failure.
  - Use of high-level libraries allowing matrix operations, neural
    network operations, differentiation operations, linear algebra operations
    etc is forbidden and will result in failure.
    (Low-level math operations: sqrt, exp, log, rand... and libraries
    like `<algorithm>` or `<iostream>` are fine)

### DATASET
Fashion MNIST (https://arxiv.org/pdf/1708.07747.pdf) a modern version of a
well-known MNIST (http://yann.lecun.com/exdb/mnist/). It is a dataset of
Zalando's article images ‒ consisting of a training set of 60,000 examples
and a test set of 10,000 examples. Each example is a 28x28 grayscale image,
associated with a label from 10 classes. The dataset is in CSV format. There
are four data files included:
 - `fashion_mnist_train_vectors.csv` - training input vectors
 - `fashion_mnist_test_vectors.csv` - testing input vectors
 - `fashion_mnist_train_labels.csv` - training labels
 - `fashion_mnist_test_labels.csv` - testing labels

### REMARKS
  - You can work alone or you can make teams of two.
  - You may or may not include the datasets in your .zip file, we will be
    replacing them. Either way **keep the `data` folder in your solution**
    and load all the datasets from it.
  - What you do internally with the training dataset is up to you.
  - Write reasonable doc-strings.
  - Ask questions in the Discussion forum.

### Aisa
  - `.exe` files are not runnable on Aisa.
  - Aisa runs on Red Hat Enterprise Linux version 7
  - Aisa has 4×16 cores, OpenMP or similar easy parallelism may help (in case
    you use it, please leave some cores to other applications).
  - Be considerate to others and run your network on Aisa with decreased priority
    using for example the `nice` program. (especially if you are using multiple cores)
    (more info here: https://www.fi.muni.cz/tech/unix/computation.html.en)
  - If you are having a problem with missing compilers/maven on Aisa, you can
    add such tools by adding modules
    (https://www.fi.muni.cz/tech/unix/modules.html.en). Please, do note, that
    if your implementation requires such modules, your `run.sh` script must include
    them as well, otherwise, the `run.sh` script won't work. Make sure your solution
    does not require modules which you are adding in your `.bashrc` file and
    which are not present in the `run.sh` file.
  - (mainly) Windows users
    - Test your submission from the same zip file you are submitting to IS, not just
      the cloned repository. Github automatically removes carriage return characters
      (Windows new line) on Linux machines, but your zip submission might still
      contain them, which means the RUN file fails. We are trying to mitigate this
      by deleting them, but there are no guarantees that it will work.
    - you don't have to worry about the execution permissions required for the `run.sh`
      script, we add them automatically

### TIPS
  - Do not wait until the week (or even month) before the deadline!
  - Test your `run.sh` script from your .zip file on Aisa before your
    submission. Missing or non-functional `run.sh` script cannot be evaluated.
  - Do NOT shuffle testing data. It won't fit expected predictions.
  - Try to solve the XOR problem first. It is a benchmark for your training
    algorithm because it is non-linear and requires at least one hidden layer.
    The presented network solving XOR in the lecture is minimal and it can be hard
    to find, so use more neurons in the hidden layer. If you can't solve
    the XOR problem, you can't solve Fashion-MNIST.
  - Reuse memory. You are implementing an iterative process, so don't allocate
    new vectors and matrices all the time. An immutable approach is nice but
    very inappropriate. Don't make unnecessary copies.
  - Objects are fine, but be careful about the depth of object hierarchy you
    are going to create. Always remember that you are trying to be fast.
  - Double precision is fine. You may try to use floats. Do not use BigDecimals
    or any other high precision objects.
  - Dont forget to use compiler optimizations (e.g. -O3 or -Ofast)
  - Simple SGD is most likely not fast enough, you are going to need to
    implement some more advanced features as well (or maybe not, but it's highly
    recommended). You can add things like momentum, weight decay, dropout or you
    can try to use some advanced optimizers like RMSProp, AdaGrad or Adam.
  - Start with smaller networks and increase network topology carefully.
  - Consider validation of the model using part of the **training** dataset.
  - Adjust hyperparameters to increase your internal validation accuracy.
  - DO NOT WAIT UNTIL THE WEEK BEFORE THE DEADLINE!

### FAQ
**Q:** Can I write in Python, please, please, pretty please?
**A:** No. It's too slow without matrix libraries.
 
**Q:** Can I implement a convolutional neural network instead of the feed-forward
    neural network?
**A:** Yes, but it might be harder.

**Q:** Can I use attention?
**A:** Yes, but it might be much harder.Good afternoon,
the following is the task description for your projects.
If you have any questions, please ask in the discussion forums.

## PV021 project -- Deep Learning from Scratch

### DEADLINE
Thursday 1. 12. 2022 23:59 (December 1st)

### TASK
  - Implement a neural network in a low-level programming language (C/C++/Java/Rust)
    without the use of advanced libraries or frameworks.
  - Train it on the supplied Fashion-MNIST dataset using a backpropagation
   algorithm.

### REQUIREMENTS
  - Your solution must follow the project template (will be provided
    in the study materials) and must be runnable on the Aisa server.
  - Your solution must achieve at least 88% accuracy on the test set.
  - Your solution must finish within 30 minutes.
    (parse inputs, train, evaluate, export results.)
  - Your solution must contain a runnable script called `run.sh` (not `run`, not
   `RUN.sh`, not `run.exe` etc.) which compiles and executes your code
    (and exports the results).
  - Your solution must output two files to the root project directory:
    (next to `example_test_predictions.csv` file):
     - `train_predictions.csv` - your network predictions for the train set.
     - `test_predictions.csv` - your network predictions for the test set.
  - The format of these files has to be the same as the supplied
    training/testing labels:
     - One prediction per line.
     - Prediction for i-th input vector (ordered by the input .csv file) must
       be on i-th line in the associated output file.
     - Each prediction is a single integer 0 - 9.
  - Replace the first line of this file with UČOs and names
  - Submit your solution in a .zip format before the deadline to a vault in IS.
    We will let you know once we create it.

### SCORING
  - The project has binary scoring -- either you pass it or fail it.
    It is required to pass it in order to get your mark from the oral exam
    and pass the subject.
  - All submitted source files will be checked manually.
  - All submitted solutions will be also checked by an automatic evaluator.
    A simple evaluator is provided in the project template as well.
  - All submitted solutions will be checked for plagiarism.
    Submitting someone else's solution (e.g. publicly available solutions)
    as your own will result in failure and will lead to disciplinary action.
  - Any implementation that uses testing input vectors for anything else
    than final evaluation will result in failure.
  - Use of high-level libraries allowing matrix operations, neural
    network operations, differentiation operations, linear algebra operations
    etc is forbidden and will result in failure.
    (Low-level math operations: sqrt, exp, log, rand... and libraries
    like `<algorithm>` or `<iostream>` are fine)

### DATASET
Fashion MNIST (https://arxiv.org/pdf/1708.07747.pdf) a modern version of a
well-known MNIST (http://yann.lecun.com/exdb/mnist/). It is a dataset of
Zalando's article images ‒ consisting of a training set of 60,000 examples
and a test set of 10,000 examples. Each example is a 28x28 grayscale image,
associated with a label from 10 classes. The dataset is in CSV format. There
are four data files included:
 - `fashion_mnist_train_vectors.csv` - training input vectors
 - `fashion_mnist_test_vectors.csv` - testing input vectors
 - `fashion_mnist_train_labels.csv` - training labels
 - `fashion_mnist_test_labels.csv` - testing labels

### REMARKS
  - You can work alone or you can make teams of two.
  - You may or may not include the datasets in your .zip file, we will be
    replacing them. Either way **keep the `data` folder in your solution**
    and load all the datasets from it.
  - What you do internally with the training dataset is up to you.
  - Write reasonable doc-strings.
  - Ask questions in the Discussion forum.

### Aisa
  - `.exe` files are not runnable on Aisa.
  - Aisa runs on Red Hat Enterprise Linux version 7
  - Aisa has 4×16 cores, OpenMP or similar easy parallelism may help (in case
    you use it, please leave some cores to other applications).
  - Be considerate to others and run your network on Aisa with decreased priority
    using for example the `nice` program. (especially if you are using multiple cores)
    (more info here: https://www.fi.muni.cz/tech/unix/computation.html.en)
  - If you are having a problem with missing compilers/maven on Aisa, you can
    add such tools by adding modules
    (https://www.fi.muni.cz/tech/unix/modules.html.en). Please, do note, that
    if your implementation requires such modules, your `run.sh` script must include
    them as well, otherwise, the `run.sh` script won't work. Make sure your solution
    does not require modules which you are adding in your `.bashrc` file and
    which are not present in the `run.sh` file.
  - (mainly) Windows users
    - Test your submission from the same zip file you are submitting to IS, not just
      the cloned repository. Github automatically removes carriage return characters
      (Windows new line) on Linux machines, but your zip submission might still
      contain them, which means the RUN file fails. We are trying to mitigate this
      by deleting them, but there are no guarantees that it will work.
    - you don't have to worry about the execution permissions required for the `run.sh`
      script, we add them automatically

### TIPS
  - Do not wait until the week (or even month) before the deadline!
  - Test your `run.sh` script from your .zip file on Aisa before your
    submission. Missing or non-functional `run.sh` script cannot be evaluated.
  - Do NOT shuffle testing data. It won't fit expected predictions.
  - Try to solve the XOR problem first. It is a benchmark for your training
    algorithm because it is non-linear and requires at least one hidden layer.
    The presented network solving XOR in the lecture is minimal and it can be hard
    to find, so use more neurons in the hidden layer. If you can't solve
    the XOR problem, you can't solve Fashion-MNIST.
  - Reuse memory. You are implementing an iterative process, so don't allocate
    new vectors and matrices all the time. An immutable approach is nice but
    very inappropriate. Don't make unnecessary copies.
  - Objects are fine, but be careful about the depth of object hierarchy you
    are going to create. Always remember that you are trying to be fast.
  - Double precision is fine. You may try to use floats. Do not use BigDecimals
    or any other high precision objects.
  - Dont forget to use compiler optimizations (e.g. -O3 or -Ofast)
  - Simple SGD is most likely not fast enough, you are going to need to
    implement some more advanced features as well (or maybe not, but it's highly
    recommended). You can add things like momentum, weight decay, dropout or you
    can try to use some advanced optimizers like RMSProp, AdaGrad or Adam.
  - Start with smaller networks and increase network topology carefully.
  - Consider validation of the model using part of the **training** dataset.
  - Adjust hyperparameters to increase your internal validation accuracy.
  - DO NOT WAIT UNTIL THE WEEK BEFORE THE DEADLINE!

### FAQ
**Q:** Can I write in Python, please, please, pretty please?

**A:** No. It's too slow without matrix libraries.
 
**Q:** Can I implement a convolutional neural network instead of the feed-forward
    neural network?
    
**A:** Yes, but it might be harder.

**Q:** Can I use attention?

**A:** Yes, but it might be much harder.
