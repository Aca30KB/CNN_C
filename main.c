#define EXTERN

#include "cnn.h"
#include "image.h"
#include "data.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

int epochs;
int trainSize;
int testSize;

int main(void)
{
    CNN *cnn;
    IMAGE *testImages;
    IMAGE *trainImages;
    DATA *trainData;
    DATA *testData;

    srand((unsigned int)time(NULL));

    trainSize = 60000;
    testSize = 10000;
    epochs = 10;

    printf("Broj epoha: %d\n", epochs);

    trainData = (DATA *)calloc(1UL, sizeof(DATA));
    trainData->filename = "mnist_train.csv";
    trainData->nbRecords = trainSize;

    testData = (DATA *)calloc(1UL, sizeof(DATA));
    testData->filename = "mnist_test.csv";
    testData->nbRecords = testSize;

    trainImages = (IMAGE *)calloc((size_t)trainData->nbRecords, sizeof(IMAGE));
    testImages = (IMAGE *)calloc((size_t)testData->nbRecords, sizeof(IMAGE));

    allocData(trainImages, trainSize);
    allocData(testImages, testSize);

    printf("Starting load data..\n");
    loadData(trainData, trainImages);
    printf("Loaded train data...\n");
    loadData(testData, testImages);
    printf("Loaded test data...\n");

    cnn = (CNN *)calloc(1UL, sizeof(CNN));
    allocCNN(cnn);
    initweights(cnn);

    printf("Percentage of success before training: %f\n", test(cnn, testImages));

    backPropLearning(cnn, trainImages);

    printf("Percentage of success after training: %f\n", test(cnn, testImages));
    freeCNN(cnn);
    free(cnn);

    freeData(trainImages, trainSize);
    freeData(testImages, testSize);

    free(trainImages);
    free(testImages);

    free(trainData);
    free(testData);

    return 0;
}
