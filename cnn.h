#ifndef CNN_H
#define CNN_H

#ifndef EXTERN
#define EXTERN extern
#endif

#include "image.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

EXTERN gsl_rng *r;
EXTERN int epochs;

typedef struct _cnn
{
    int L;            /* broj skrivenih slojeva */
    double alpha;     /* stopa obucavanja */
    int rows;         /* broj redova ulazne slike */
    int cols;         /* broj kolona ulazne slike */
    int n;            /* duzina vektora ulazne slike (rows * cols) ujedno i ulaznog sloja */
    int J;            /* duzina vektora skrivenih slojeva */
    int K;            /* duzina vektora izlaznog sloja */
    double **w_in;    /* matrica tezina izmedju ulaznog i prvog skrivenog sloja (785 * 32) */
    double **w_in_T;  /* transponovana matrica tezina izmedju ulaznog i prvog skrivenog sloja (32 * 785) */
    double ***w;      /* niz matrica tezina u skrivenim slojevima (33 * 32) */
    double **w_out;   /* matrica tezina izmedju skrivenog sloja i izlaznog sloja (33 * 10) */
    double **w_out_T; /* transponovana matrica tezina izmedju skrivenog i izlaznog sloja */
    double *x;        /* vektor ulaznog sloja (784) */
    double **a;       /* cvorovi u skrivenim slojevima (32) */
    double *b;        /* izlazni sloj (10) */
    double **in;      /* vektor ulaznih vrednosti u slojeve mreze */
} CNN;

/*  alokacija CNN */
void allocCNN(CNN *);

/* sigmoidna funkcija */
double sigmoid(double);

/* izvod sigmoidne funkcije */
double dSigmoid(double);

/* ispravljacka jedinica */
double ReLU(double);

/* izvod ispravljacke jedinice */
double dReLU(double);

/* inicijalizacija tezina */
void initWeights(CNN *);

double softmaxsum(double *);

void softmax(double *, double *);

int maxindex(double *);

void initweights(CNN *);

double dot(double *, double *, int);

void backPropLearning(CNN *, IMAGE *);

void onehot(double **, IMAGE *, int);

int pogodi(CNN *, IMAGE);

float test(CNN *, IMAGE *);

void transpose(double **, double **, int, int);

void freeCNN(CNN *);

#endif /* CNN_H */