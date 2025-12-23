#ifndef CNN_H
#define CNN_H

#ifndef EXTERN
#define EXTERN extern
#endif

#include "image.h"
#include <stdlib.h>
#include <math.h>

/* globalne promenljive */
EXTERN int epochs;
EXTERN int trainSize;
EXTERN int testSize;

typedef struct _cnn
{
    double alpha; /* stopa obucavanja */
    int rows;     /* broj redova ulazne slike */
    int cols;     /* broj kolona ulazne slike */
    int n;        /* duzina vektora ulazne slike (rows * cols) */
    int J;        /* broj neurona skrivenog sloja */
    int K;        /* broj neurona izlaznog sloja */

    double **w_in;  /* tezine ulaz -> skriveni sloj: (n + 1) x J (bias u prvom redu) */
    double **w_out; /* tezine skriveni sloj -> izlaz: (J + 1) x K (bias u prvom redu) */

    double *x;    /* ulazni vektor (n) */
    double *in_h; /* ulaz u skriveni sloj (J) */
    double *a;    /* aktivacije skrivenog sloja (J) */
    double *in_o; /* ulaz u izlazni sloj (K) */
    double *b;    /* izlazni sloj (K) */
} CNN;

/* alokacija strukture CNN (unutrasnje polje) */
void allocCNN(CNN *cnn);

/* oslobadjanje memorije CNN (unutrasnje polje) */
void freeCNN(CNN *cnn);

/* inicijalizacija tezina */
void initweights(CNN *cnn);

/* sigmoidna funkcija */
double sigmoid(double x);

/* izvod sigmoidne funkcije */
double dSigmoid(double x);

/* ReLU */
double ReLU(double x);

/* izvod ReLU (leaky) */
double dReLU(double x);

/* stabilan softmax (radi na vektoru b, rezultat u y) */
void softmax(const double *b, double *y, int K);

/* skalarni proizvod sa bias tezinskim elementom w[0] */
double dot(const double *w, const double *a, int size);

/* one-hot kodiranje labela */
void onehot(double **y, IMAGE *images, int size, int K);

/* ucenje mreze (SGD) */
void backPropLearning(CNN *cnn, IMAGE *train_images);

/* predikcija jedne slike */
int pogodi(CNN *cnn, IMAGE image);

/* testiranje tacnosti na zbirci slika */
float test(CNN *cnn, IMAGE *images);

#endif /* CNN_H */
