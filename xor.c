#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#define INPUT_SIZE 2
#define HIDDEN_SIZE 2
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.17
#define EPOCHS 400

// LFSR parameters for generating pseudo-random numbers
#define LFSR_WIDTH 16
#define LFSR_SEED 0xACE1u

// relu activation function
double relu(double x) {
    return fmax(0, x);
}

// Derivative of the relu function
double relu_derivative(double x) {
    if (x > 0) {
        return 1;
    } else {
        return 0;
    }
}

// LFSR-based random number generator
unsigned int lfsr = LFSR_SEED;
unsigned int lfsr_rand() {
    lfsr = (lfsr >> 1) ^ (-(lfsr & 1u) & 0xB400u);
    return lfsr & 0xFFFFu;
}

int main() {
    // XOR input data and corresponding target output
    double input_data[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double target_output[4] = {0, 1, 1, 0};

    // Initialize random weights and biases
    double hidden_weights[INPUT_SIZE][HIDDEN_SIZE];
    double hidden_bias[HIDDEN_SIZE];
    double hidden_layer[HIDDEN_SIZE];
    double output_weights[HIDDEN_SIZE][OUTPUT_SIZE];
    double output_bias[OUTPUT_SIZE];
    double output_layer[OUTPUT_SIZE];

    // Initialize weights and biases using LFSR-based random numbers
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden_weights[i][j] = (lfsr_rand() / (double)0xFFFF) * 2 - 1;
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_bias[i] = (lfsr_rand() / (double)0xFFFF) * 2 - 1;
        output_bias[0] = (lfsr_rand() / (double)0xFFFF) * 2 - 1;
    }
     for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden_weights[i][j] = (float)rand() / RAND_MAX;
        }
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_bias[i] =  (float)rand() / RAND_MAX;
        output_bias[0] =  (float)rand() / RAND_MAX;
    }

    // Training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double total_error = 0.0;

        for (int example = 0; example < 4; example++) {
            // Forward pass
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                hidden_layer[j] = 0;
                for (int i = 0; i < INPUT_SIZE; i++) {
                    hidden_layer[j] += input_data[example][i] * hidden_weights[i][j];
                }
                hidden_layer[j] += hidden_bias[j];
                hidden_layer[j] = relu(hidden_layer[j]);
            }

            double predicted_output = 0;
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                predicted_output += hidden_layer[j] * output_weights[j][0];
            }
            predicted_output += output_bias[0];
            

            // Calculate the error
            double error = target_output[example] - predicted_output;
            total_error += 0.5 * pow(error, 2);

            // Backpropagation
            double output_delta = error * relu_derivative(predicted_output);

            for (int j = 0; j < HIDDEN_SIZE; j++) {
                double hidden_error = output_delta * output_weights[j][0];
                double hidden_delta = hidden_error * relu_derivative(hidden_layer[j]);

                // Update output weights
                output_weights[j][0] += LEARNING_RATE * hidden_layer[j] * output_delta;

                // Update hidden weights
                for (int i = 0; i < INPUT_SIZE; i++) {
                    hidden_weights[i][j] += LEARNING_RATE * input_data[example][i] * hidden_delta;
                }

                // Update biases
                output_bias[0] += LEARNING_RATE * output_delta;
                hidden_bias[j] += LEARNING_RATE * hidden_delta;
            }
        }

        // Print the total error for this epoch
        if (epoch % 100 == 0) {
            printf("Epoch %d - Total Error: %lf\n", epoch, total_error);
        }
    }

    // Testing and Evaluation
    int correct_predictions = 0;
    for (int example = 0; example < 4; example++) {
        // Forward pass for testing
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            hidden_layer[j] = 0;
            for (int i = 0; i < INPUT_SIZE; i++) {
                hidden_layer[j] += input_data[example][i] * hidden_weights[i][j];
            }
            hidden_layer[j] += hidden_bias[j];
            hidden_layer[j] = relu(hidden_layer[j]);
        }

        double predicted_output = 0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            predicted_output += hidden_layer[j] * output_weights[j][0];
        }
        predicted_output += output_bias[0];
        predicted_output = relu(predicted_output);

        // Calculate the binary prediction (0 or 1)
        int binary_prediction = (predicted_output > 0.5) ? 1 : 0;

        printf("Input: %lf, %lf, Predicted Output: %lf, Predicted Class: %d, Ground Truth: %d\n", input_data[example][0], input_data[example][1], predicted_output, binary_prediction, (int)target_output[example]);

        // Check if the prediction matches the ground truth
        if (binary_prediction == (int)target_output[example]) {
            correct_predictions++;
        }
    }

    // Calculate accuracy
    double accuracy = (double)correct_predictions / 4.0; // Assuming you have 4 test examples

    // Print the accuracy
    printf("Accuracy: %.2f\n", accuracy);

    return 0;
}
