#include<stdio.h>
#include<dirent.h>
#include<stdlib.h>
#include<string.h>
#include<sys/stat.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include"lexer.h"



#define InputNodes 784
#define HiddenNodes 120
#define Outputs 9
extern double HiddenLayerBias[HiddenNodes];
extern double OuputLayerBias[Outputs];


extern double HiddenWeights[InputNodes][HiddenNodes];
extern double OutputWeights[HiddenNodes][Outputs];


int dir_exists(char* path){
	if(mkdir(path,0777)==-1){
		return 0;
	}
	return 1;
}


void read_hiddenweights(){
	FILE* file;
	int line=0;
	file=fopen("bin/HiddenWeights.neural","r");
	while(!feof(file)){
			for(int i=0; i<HiddenNodes; i++){
				if(fscanf(file, "%lf",&HiddenWeights[line][i])==EOF)
				 	break;
			}
			line++;
			if(line==InputNodes) break;
	}
	fclose(file);
}

void read_outputweights(){
	FILE* file;
	int line=0;
	file=fopen("bin/OutputWeights.neural","r");
	while(!feof(file)){
			for(int i=0; i<Outputs; i++){
				if(fscanf(file, "%lf",&OutputWeights[line][i])==EOF)
				 	break;
			}
			line++;
			if(line==HiddenNodes) break;
	}
	fclose(file);
}

void read_hiddenbias(){
	FILE* file;
	file=fopen("bin/hiddenbias.neural", "r");
	for(int i =0; i<HiddenNodes; i++){
		if(fscanf(file, "%lf", &HiddenLayerBias[i])==EOF)
				break;
	}
}

void read_outputbias(){
	FILE* file;
	file=fopen("bin/outputbias.neural", "r");
	for(int i =0; i<Outputs; i++){
		if(fscanf(file, "%lf", &OuputLayerBias[i])==EOF)
				break;
	}
}
//____________________________write____________________

void write_hiddenweights(){
	FILE* writer= fopen("bin/HiddenWeights.neural", "w");
	for(int i=0; i<InputNodes; i++){
		int j =0;
		while(j<HiddenNodes-1){
			fprintf(writer, "%lf",HiddenWeights[i][j]);
			fprintf(writer," ");
			j++;
		}
		fprintf(writer, "%lf",HiddenWeights[i][HiddenNodes-1]);
		fprintf(writer,"\n");
	}
}

void write_outputweights(){
	FILE* writer= fopen("bin/OutputWeights.neural", "w");
	for(int i=0; i<HiddenNodes; i++){
		int j =0;
		while(j<Outputs-1){
			fprintf(writer, "%lf",OutputWeights[i][j]);
			fprintf(writer," ");
			j++;
		}
		fprintf(writer, "%lf",OutputWeights[i][Outputs-1]);
		fprintf(writer,"\n");
	}
}


void write_hiddenbias(){
	FILE* writer= fopen("bin/hiddenbias.neural", "w");
	for(int i =0; i<HiddenNodes; i++){
		fprintf(writer, "%lf", HiddenLayerBias[i]);
		fprintf(writer," ");
	}
}

void write_outputbias(){
	FILE* writer= fopen("bin/outputbias.neural", "w");
	for(int i =0; i<Outputs; i++){
		fprintf(writer, "%lf", OuputLayerBias[i]);
		fprintf(writer," ");
	}
}


void print_matrix(double matrix[][10]){
	for(size_t i =0; i< 2; i++){
		for(size_t j=0; j<10; j++){
			printf("%f ", matrix[i][j]);
		}
		printf("\n");
	}
}

