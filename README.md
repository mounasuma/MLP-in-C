# MLP-in-C
The code defines and structures the MLP using the C programming language. It includes the creation of layers, neurons, and the connections between them, effectively building a neural network tailored for performing xor calculations.
This project implements a Multilayer Perceptron (MLP) to perform xor calculations using the C programming language in Visual Studio Code.

#Prerequisites
To compile and run this code in Visual Studio Code, you will need:

Visual Studio Code (or your preferred code editor).
A C/C++ extension for Visual Studio Code.
A C compiler (e.g., GCC).
Standard C libraries.
C Standard Library Headers.
How to Compile and Run
Clone or download the repository to your local machine.

Open the project folder in Visual Studio Code.

Open the C source code file (main.c) in the code editor.

Configure the project settings in Visual Studio Code to use your preferred C compiler and build system.

Build the code using the integrated tools in Visual Studio Code or using terminal commands. For example, using GCC:

Run the compiled program within Visual Studio Code or through the terminal:

#Code Description
main.c contains the C code for the MLP implementation, including network architecture, activation functions, training, and testing.

The code employs LFSR-based random number generation for initializing weights and biases.

Training involves backpropagation and weight updates with a defined learning rate.

After training, the code tests the network's performance and calculates accuracy.

#Configuration and Parameters
You can configure the following parameters in the code:

INPUT_SIZE: The number of input neurons.
HIDDEN_SIZE: The number of neurons in the hidden layer.
OUTPUT_SIZE: The number of output neurons.
LEARNING_RATE: The learning rate for weight updates.
EPOCHS: The number of training epochs.
#Acknowledgments
The code utilizes a RELU activation function for the MLP.

It also includes the use of LFSR-based pseudo-random number generation for initializing weights and biases.

#Author
MOUNA SUMA MANDAVA

#Date
26/10/2023
