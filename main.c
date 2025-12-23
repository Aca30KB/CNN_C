#define EXTERN

#include "cnn.h"
#include "image.h"
#include "data.h"
#include <unistd.h>
#include <stdlib.h>

int main(void);
int main()
{

    CNN *cnn;
    IMAGE *testImages;
    IMAGE *trainImages;
    DATA *trainData;
    DATA *testData;
    const gsl_rng_type *T;
    /*
        if (argc >= 2)
        {
            fprintf(stderr, "Proverite argumente komandne linije!\n");
            exit(EXIT_FAILURE);
        }*/

    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);
    gsl_rng_set(r, (unsigned long)getpid());

    trainSize = 60000;
    testSize = 10000;
    epochs = 5;

    fprintf(stdout, "%d\n", epochs);

    trainData = (DATA *)calloc(1UL, sizeof(DATA));
    trainData->filename = "mnist_train.csv";
    trainData->nbRecords = 60000;

    testData = (DATA *)calloc(1UL, sizeof(DATA));
    testData->filename = "mnist_test.csv";
    testData->nbRecords = 10000;

    testImages = (IMAGE *)calloc((size_t)testData->nbRecords, sizeof(IMAGE));
    trainImages = (IMAGE *)calloc((size_t)trainData->nbRecords, sizeof(IMAGE));

    allocData(testImages, testSize);
    allocData(trainImages, trainSize);

    printf("Pocetak ucitavanja podataka...\n");

    loadData(trainData, trainImages);
    printf("Ucitani trening podaci...\n");
    loadData(testData, testImages);
    printf("Ucitani test podaci...\n");

    cnn = (CNN *)calloc(1UL, sizeof(CNN));

    allocCNN(cnn);
    /* inicijalizacija matrica tezina */
    initweights(cnn);

    printf("Procenat uspesnosti pre treninga mreze: %f\n", test(cnn, testImages));

    backPropLearning(cnn, trainImages);

    printf("Procenat uspesnosti istrenirane mreze: %f\n", test(cnn, testImages));

    freeCNN(cnn);
    free(cnn);
    free(testImages);
    free(trainImages);
    free(trainData);
    gsl_rng_free(r);
    freeData(&trainImages, trainSize);
    freeData(&testImages, testSize);

    exit(EXIT_SUCCESS);
}