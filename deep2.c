//Note: this code was written when I was 13 years old
//Now I'm 16, and this codes are like pirates code..
//However, it seems like my C programming skills back then was great.
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define INPUT_NODES 2
#define HIDDEN_NODES 5
#define OUTPUT_NODES 1
#define LEARNING_RATE 0.5
#define EPOCHES 10000

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid function
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Initialize weights with random values
void initialize_weights(double weights[], int size) {
    for (int i = 0; i < size; i++) {
        weights[i] = ((double)rand() / RAND_MAX) * 2 - 1; // Random values between -1 and 1
    }
}

// Mean squared error function
double calculate_mse(double output[], double targets[], int size) {
    double mse = 0.0;
    for (int i = 0; i < size; i++) {
        mse += pow(targets[i] - output[i], 2);
    }
    return mse / size;
}

int main() {
    // Input and output data (XOR problem)
    double input[4][INPUT_NODES] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    double targets[4][OUTPUT_NODES] = {
        {0},
        {1},
        {1},
        {0}
    };

    // Weight initialization
    double input_hidden_weights[INPUT_NODES * HIDDEN_NODES];
    double hidden_output_weights[HIDDEN_NODES * OUTPUT_NODES];
    double hidden_bias[HIDDEN_NODES];
    double output_bias[OUTPUT_NODES];

    initialize_weights(input_hidden_weights, INPUT_NODES * HIDDEN_NODES);
    initialize_weights(hidden_output_weights, HIDDEN_NODES * OUTPUT_NODES);
    initialize_weights(hidden_bias, HIDDEN_NODES);
    initialize_weights(output_bias, OUTPUT_NODES);

    // Training the neural network
    for (int epoch = 0; epoch < EPOCHES; epoch++) {
        double mse = 0.0;

        for (int i = 0; i < 4; i++) {
            double input_layer[INPUT_NODES];
            double hidden_layer[HIDDEN_NODES];
            double output_layer[OUTPUT_NODES];

            // Forward pass
            for (int j = 0; j < INPUT_NODES; j++) {
                input_layer[j] = input[i][j];  // Correct assignment
            }

            for (int j = 0; j < HIDDEN_NODES; j++) {
                double activation = hidden_bias[j];
                for (int k = 0; k < INPUT_NODES; k++) {
                    activation += input_layer[k] * input_hidden_weights[j * INPUT_NODES + k];
                }
                hidden_layer[j] = sigmoid(activation);  // Activation for hidden layer
            }

            for (int j = 0; j < OUTPUT_NODES; j++) {
                double activation = output_bias[j];
                for (int k = 0; k < HIDDEN_NODES; k++) {
                    activation += hidden_layer[k] * hidden_output_weights[k * OUTPUT_NODES + j];
                }
                output_layer[j] = sigmoid(activation);  // Output layer activation
            }

            // Backward pass (error calculation)
            double output_errors[OUTPUT_NODES];
            double hidden_errors[HIDDEN_NODES];

            for (int j = 0; j < OUTPUT_NODES; j++) {
                output_errors[j] = targets[i][j] - output_layer[j];
                mse += pow(output_errors[j], 2);  // Sum squared errors for MSE calculation
            }

            for (int j = 0; j < HIDDEN_NODES; j++) {
                hidden_errors[j] = 0.0;
                for (int k = 0; k < OUTPUT_NODES; k++) {
                    hidden_errors[j] += output_errors[k] * hidden_output_weights[k * OUTPUT_NODES + j];
                }
                hidden_errors[j] *= sigmoid_derivative(hidden_layer[j]);
            }

            // Update weights and biases
            for (int j = 0; j < OUTPUT_NODES; j++) {
                output_bias[j] += LEARNING_RATE * output_errors[j] * sigmoid_derivative(output_layer[j]);
                for (int k = 0; k < HIDDEN_NODES; k++) {
                    hidden_output_weights[k * OUTPUT_NODES + j] += LEARNING_RATE * output_errors[j] * hidden_layer[k];
                }
            }

            for (int j = 0; j < HIDDEN_NODES; j++) {
                hidden_bias[j] += LEARNING_RATE * hidden_errors[j];
                for (int k = 0; k < INPUT_NODES; k++) {
                    input_hidden_weights[j * INPUT_NODES + k] += LEARNING_RATE * hidden_errors[j] * input_layer[k];
                }
            }
        }

        // Print the MSE every 1000 epochs for monitoring
        if (epoch % 1000 == 0) {
            mse /= 4;  // Calculate average MSE over the 4 samples
            printf("Epoch %d, MSE: %.6f\n", epoch + 1, mse);
        }
    }

    // Final output and classification
    for (int i = 0; i < 4; i++) {
        double hidden_layer[HIDDEN_NODES];
        double output_layer[OUTPUT_NODES];

        for (int j = 0; j < HIDDEN_NODES; j++) {
            double activation = hidden_bias[j];
            for (int k = 0; k < INPUT_NODES; k++) {
                activation += input[i][k] * input_hidden_weights[j * INPUT_NODES + k];
            }
            hidden_layer[j] = sigmoid(activation);
        }

        for (int j = 0; j < OUTPUT_NODES; j++) {
            double activation = output_bias[j];
            for (int k = 0; k < HIDDEN_NODES; k++) {
                activation += hidden_layer[k] * hidden_output_weights[k * OUTPUT_NODES + j];
            }
            output_layer[j] = sigmoid(activation);
        }

        printf("input: %d, %d  model output: %f, and target: %f\n",
               (int)input[i][0], (int)input[i][1], output_layer[0], targets[i][0]);

        if (output_layer[0] < 0.5) {
            printf("This is a 0\n");
        } else {
            printf("This is a 1\n");
        }
    }

    // Print the final weights (for debugging)
    printf("Trained weights and biases: \n");
    for (int i = 0; i < INPUT_NODES * HIDDEN_NODES; i++) {
        printf("input_hidden_weights[%d] = %f\n", i, input_hidden_weights[i]);
    }
    for (int i = 0; i < HIDDEN_NODES * OUTPUT_NODES; i++) {
        printf("hidden_output_weights[%d] = %f\n", i, hidden_output_weights[i]);
    }
    for (int i = 0; i < HIDDEN_NODES; i++) {
        printf("hidden_bias[%d] = %f\n", i, hidden_bias[i]);
    }
    for (int i = 0; i < OUTPUT_NODES; i++) {
        printf("output_bias[%d] = %f\n", i, output_bias[i]);
    }

    return 0;
}

