#include <stdlib.h>
#include <stdio.h>
#include<math.h>

double init_weights(){
    // printf("k");
    return ( (double)rand())/((double)RAND_MAX); 
}
double sigmoid(double x){return 1/(1+exp(-x));}
double dsigmoid(double x){return x*(1-x);}


void shuffle(int *array, size_t n){
    if(n>1){
        size_t i;
        for(i=0; i<n-1; i++){
            size_t j =i+ rand()/(RAND_MAX/(n-i)+1);
            int t =array[j];
            array[j]=array[i];
            array[i]=t;
        }
    }
}
#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4



int main(void){
    const double lr =0.1f;

    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];

    double hiddenLayerBias[numHiddenNodes];
    double ouputLayerBias[numOutputs];


    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numInputs];
    double training_inputs[numTrainingSets][numInputs]=
                                                        {
                                                            {0.0f, 0.0f},
                                                            {1.0f, 0.0f},
                                                            {0.0f, 1.0f},
                                                            {1.0f, 1.0f},
                                                        };
    
    double training_outputs[numTrainingSets][numOutputs]={{0.0f},
                                                          {1.0f},
                                                          {1.0f},
                                                          {0.0f}
                                                            };
    
    // printf("OK1\n");
    for(int i =0; i<numInputs; i++){
        for(int j=0; j<numHiddenNodes;j++){
            hiddenWeights[i][j]=init_weights();
        }
    }
    // printf("OK2\n");

    for(int i =0; i<numHiddenNodes; i++){
        for(int j=0; j<numOutputs;j++){
            outputWeights[i][j]=init_weights();
        }
    }

    for(int i=0; i <numHiddenNodes; i++){
        hiddenLayerBias[i]=init_weights();
    }

    for(int i=0; i <numOutputs; i++){
        ouputLayerBias[i]=init_weights();
    }

    int trainingSetOrder[]={0,1,2,3};

    int numberOfEpochs=10000;

    //train
    for(int epoch=0; epoch<numberOfEpochs; epoch++){
        // printf("loop\n");
        shuffle(trainingSetOrder, numTrainingSets);

        for(int x=0; x<numTrainingSets; x++){
            int i=trainingSetOrder[x];

            //forward pass

            //hidden layer
            for(int j=0; j<numHiddenNodes; j++){
                double activation=hiddenLayerBias[j];
                for(int k =0; k<numInputs; k++){
                    activation+=training_inputs[i][k] * hiddenWeights[k][j];
                }
                hiddenLayer[j] = sigmoid(activation);
            }

            //output layer
            for(int j=0; j<numOutputs; j++){
                double activation=ouputLayerBias[j];
                for(int  k=0; k<numHiddenNodes; k++){
                    activation+=hiddenLayer[k] * outputWeights[k][j];
                }
                outputLayer[j] = sigmoid(activation);
            }
            printf("Input: %g  %g Output: %g Predicted Output: %g \n",
                    training_inputs[i][0],training_inputs[i][1], outputLayer[0], training_outputs[i][0]);
            //Backprop
            //change in output weights
            double deltaOutput[numOutputs];
            for(int j=0; j<numOutputs; j++){
                double error = (training_outputs[i][j]-outputLayer[j]);
                deltaOutput[j]=error * dsigmoid(outputLayer[j]);
            }
            //change in hidde weights
            double deltaHidden[numHiddenNodes];
            for(int j=0; j<numHiddenNodes; j++){
                double error=0.0f;
                for(int k =0; k<numOutputs; k++){
                    error+=deltaOutput[k]*outputWeights[j][k];
                }
                deltaHidden[j]=error * dsigmoid(hiddenLayer[j]);
            }
            //apply change in output weights
            for(int j =0; j<numOutputs; j++){
                ouputLayerBias[j]+=deltaOutput[j]*lr;
                for(int k=0;k<numHiddenNodes;k++){
                    outputWeights[k][j]+=hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }
            //apply change in hidden weights
            for(int j =0; j<numHiddenNodes; j++){
                hiddenLayerBias[j]+=deltaHidden[j]*lr;
                for(int k=0;k<numInputs;k++){
                    hiddenWeights[k][j]+=training_inputs[i][k] * deltaHidden[j] * lr;
                }
            }
        }
    }
}