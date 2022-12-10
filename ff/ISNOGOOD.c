#include <stdlib.h>
#include <stdio.h>
#include<math.h>

double init_weights(){
    return ( (double)rand())/((double)RAND_MAX); 
}
double sigmoid(double x){return 1/(1+exp(-x));}
double dsigmoid(double x){return x*(1-x);}





// double softmax(double a[numHiddenNodes], int ind){
//     double sum=0;
//     for(int i =0; i<numHiddenNodes; i++)
//         sum+=a[i];
//     return a[i]/sum;
// }






double relu(double x){
    if(x>0)
        return x;
    return 0.0f;
}
double drelu(double x){
    if(x>0)
        return 1;
    return 0.0f;
}
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




const double lr =0.2f;

double hiddenLayer[numHiddenNodes];
double outputLayer[numOutputs];

double hiddenLayerBias[numHiddenNodes];
double ouputLayerBias[numOutputs];


double hiddenWeights[numInputs][numHiddenNodes];
double outputWeights[numHiddenNodes][numOutputs];

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




// void setup(double inputs[numInputs], double outputs[numOutputs] ){
    
// }





int main(void){
    
    
    for(int i =0; i<numInputs; i++){
        for(int j=0; j<numHiddenNodes;j++){
            hiddenWeights[i][j]=init_weights();
        }
    }

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
            int w=trainingSetOrder[x];//LPOKOKPOKJPIJOJOIJ
            double inputs[numInputs];
            double outputs[numOutputs];
            for(int jind=0; jind<numInputs; jind++){
                inputs[jind]=training_inputs[w][jind];
            }
            for(int jind=0; jind<numOutputs; jind++){
                outputs[jind]=training_inputs[w][jind];
            }
            //forward pass
            //hidden layer
            for(int i=0; i<numHiddenNodes; i++){
                double activation=hiddenLayerBias[i];
                for(int j =0; j<numInputs; j++){
                    activation+=inputs[j] * hiddenWeights[j][i];
            }               
                hiddenLayer[i] = sigmoid(activation);
            }
            //output layer
            for(int i=0; i<numOutputs; i++){
                double activation=ouputLayerBias[i];
                for(int  j=0; j<numHiddenNodes; j++){
                    activation+=hiddenLayer[j] * outputWeights[j][i];
                }
                outputLayer[i] = sigmoid(activation);
            }
            printf("Input: %g  %g Output: %g Predicted Output: %g \n",
                    inputs[0],inputs[1], outputLayer[0], outputs[0]);
            //-------------------------------------------BACKPROP-----------------------------------------//
            //change in output weights
            double deltaOutput[numOutputs];
            for(int i=0; i<numOutputs; i++){
                double error = (outputs[i]-outputLayer[i]);
                deltaOutput[i]=error * sigmoid(outputLayer[i]);
            }
            //transpose output weights
            double ThiddenOutput[numOutputs][numHiddenNodes];
            for(int i =0; i<numHiddenNodes; i++ ){
                for(int  j =0; j<numOutputs; j++){
                    ThiddenOutput[j][i]=outputWeights[i][j];
                }
            }
            //change in hidden weights
            double deltaHidden[numHiddenNodes];
            for(int i=0; i<numHiddenNodes; i++){
                double error=0.0f;
                for(int j =0; j<numOutputs; j++){
                    error+=deltaOutput[j]*ThiddenOutput[j][i];
                }
                deltaHidden[i]=error * sigmoid(hiddenLayer[i]);
            }
            //apply change in output weights
            for(int i =0; i<numOutputs; i++){
                ouputLayerBias[i]+=deltaOutput[i]*lr;
                for(int j=0;j<numHiddenNodes;j++){
                    outputWeights[j][i]+=hiddenLayer[j] * deltaOutput[i] * lr;
                }
            }
            //apply change in hidden weights
            for(int i =0; i<numHiddenNodes; i++){
                hiddenLayerBias[i]+=deltaHidden[i]*lr;
                for(int j=0;j<numInputs;j++){
                    hiddenWeights[i][j]+=inputs[j] * deltaHidden[i] * lr;
                }
            }
            
        }
    }
}