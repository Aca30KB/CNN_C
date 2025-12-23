#include "cnn.h"
#include <assert.h>
#include <stdio.h>

static double leak = 0.01;
static double sigma = 0.1;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Gaussian generator (Box–Muller) bez GSL-a */
static double rand_normal(double mean, double stddev)
{
    double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);

    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return z0 * stddev + mean;
}

void allocCNN(CNN *cnn)
{
    int j, n;

    cnn->alpha = 0.1; /* stopa ucenja */
    cnn->rows = 28;
    cnn->cols = 28;
    cnn->n = cnn->rows * cnn->cols; /* 784 */
    cnn->J = 128;                   /* skriveni sloj */
    cnn->K = 10;                    /* broj klasa */

    cnn->x = (double *)calloc((size_t)cnn->n, sizeof(double));
    assert(cnn->x != NULL);

    cnn->in_h = (double *)calloc((size_t)cnn->J, sizeof(double));
    assert(cnn->in_h != NULL);

    cnn->a = (double *)calloc((size_t)cnn->J, sizeof(double));
    assert(cnn->a != NULL);

    cnn->in_o = (double *)calloc((size_t)cnn->K, sizeof(double));
    assert(cnn->in_o != NULL);

    cnn->b = (double *)calloc((size_t)cnn->K, sizeof(double));
    assert(cnn->b != NULL);

    cnn->w_in = (double **)calloc((size_t)(cnn->n + 1), sizeof(double *));
    assert(cnn->w_in != NULL);
    for (n = 0; n < cnn->n + 1; n++)
    {
        cnn->w_in[n] = (double *)calloc((size_t)cnn->J, sizeof(double));
        assert(cnn->w_in[n] != NULL);
    }

    cnn->w_out = (double **)calloc((size_t)(cnn->J + 1), sizeof(double *));
    assert(cnn->w_out != NULL);
    for (j = 0; j < cnn->J + 1; j++)
    {
        cnn->w_out[j] = (double *)calloc((size_t)cnn->K, sizeof(double));
        assert(cnn->w_out[j] != NULL);
    }
}

void freeCNN(CNN *cnn)
{
    int j, n;

    free(cnn->x);
    free(cnn->in_h);
    free(cnn->a);
    free(cnn->in_o);
    free(cnn->b);

    for (n = 0; n < cnn->n + 1; n++)
        free(cnn->w_in[n]);
    free(cnn->w_in);

    for (j = 0; j < cnn->J + 1; j++)
        free(cnn->w_out[j]);
    free(cnn->w_out);
}

void initweights(CNN *cnn)
{
    int k, j1, n;

    /* He inicijalizacija */
    double scale_in = sqrt(2.0 / (double)cnn->n);
    double scale_out = sqrt(2.0 / (double)cnn->J);

    for (n = 0; n < cnn->n + 1; n++)
    {
        for (j1 = 0; j1 < cnn->J; j1++)
        {
            cnn->w_in[n][j1] = rand_normal(0.0, 1.0) * scale_in;
        }
    }

    for (j1 = 0; j1 < cnn->J + 1; j1++)
    {
        for (k = 0; k < cnn->K; k++)
        {
            cnn->w_out[j1][k] = rand_normal(0.0, 1.0) * scale_out;
        }
    }
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double dSigmoid(double x)
{
    double ex = exp(-x);
    double denom = 1.0 + ex;
    return ex / (denom * denom);
}

double ReLU(double x)
{
    return x > 0.0 ? x : 0.0;
}

double dReLU(double x)
{
    if (x <= 0.0)
        return leak;
    return 1.0;
}

void softmax(const double *b, double *y, int K)
{
    double max = b[0];
    for (int m = 1; m < K; m++)
        if (b[m] > max)
            max = b[m];

    double sum = 0.0;
    for (int m = 0; m < K; m++)
        sum += exp(b[m] - max);

    for (int m = 0; m < K; m++)
        y[m] = exp(b[m] - max) / sum;
}

double dot(const double *w, const double *a, int size)
{
    double sum = w[0]; /* bias */
    for (int i = 0; i < size; i++)
    {
        sum += w[i + 1] * a[i];
    }
    return sum;
}

void onehot(double **y, IMAGE *images, int size, int K)
{
    for (int i = 0; i < size; i++)
    {
        for (int k = 0; k < K; k++)
        {
            y[i][k] = (k == images[i].label) ? 1.0 : 0.0;
        }
    }
}

/* forward pass za jednu sliku */
static void forward(CNN *cnn)
{
    /* skriveni sloj */
    for (int j = 0; j < cnn->J; j++) {
        double sum = cnn->w_in[0][j]; // bias
        for (int p = 0; p < cnn->n; p++) {
            sum += cnn->w_in[p+1][j] * cnn->x[p];
        }
        cnn->in_h[j] = sum;
        cnn->a[j] = ReLU(sum);
    }

    /* izlazni sloj */
    for (int k = 0; k < cnn->K; k++) {
        double sum = cnn->w_out[0][k]; // bias
        for (int j = 0; j < cnn->J; j++) {
            sum += cnn->w_out[j+1][k] * cnn->a[j];
        }
        cnn->in_o[k] = sum;
    }

    softmax(cnn->in_o, cnn->b, cnn->K);
}


/* ucenje: SGD po jednom primeru */
void backPropLearning(CNN *cnn, IMAGE *train_images)
{
    int batchSize = 64;
    cnn->alpha = 0.001;  /* learning rate */

    double *delta_out = calloc(cnn->K, sizeof(double));
    double *delta_hid = calloc(cnn->J, sizeof(double));

    /* one-hot labela */
    double **y = calloc(trainSize, sizeof(double *));
    for (int i = 0; i < trainSize; i++) {
        y[i] = calloc(cnn->K, sizeof(double));
        for (int k = 0; k < cnn->K; k++) {
            y[i][k] = (k == train_images[i].label) ? 1.0 : 0.0;
        }
    }

    int *indices = malloc(trainSize * sizeof(int));
    for (int i = 0; i < trainSize; i++) indices[i] = i;

    for (int epoch = 0; epoch < epochs; epoch++) {
        /* shuffle indeksa */
        for (int i = trainSize - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
        }

        double loss = 0.0;

        for (int start = 0; start < trainSize; start += batchSize) {
            int end = (start + batchSize < trainSize) ? start + batchSize : trainSize;
            int currentBatch = end - start;

            /* gradijenti */
            double **grad_w_in = calloc(cnn->n + 1, sizeof(double *));
            for (int p = 0; p < cnn->n + 1; p++)
                grad_w_in[p] = calloc(cnn->J, sizeof(double));

            double **grad_w_out = calloc(cnn->J + 1, sizeof(double *));
            for (int j = 0; j < cnn->J + 1; j++)
                grad_w_out[j] = calloc(cnn->K, sizeof(double));

            for (int ii = start; ii < end; ii++) {
                int i = indices[ii];

                /* ucitavanje slike */
                for (int p = 0; p < cnn->n; p++)
                    cnn->x[p] = train_images[i].data[p];

                /* forward */
                forward(cnn);

                /* loss */
                int label = train_images[i].label;
                loss -= log(cnn->b[label] + 1e-9);

                /* delta izlaz */
                for (int k = 0; k < cnn->K; k++)
                    delta_out[k] = cnn->b[k] - y[i][k];

                /* delta skriveni sloj */
                for (int j = 0; j < cnn->J; j++) {
                    double sum = 0.0;
                    for (int k = 0; k < cnn->K; k++)
                        sum += cnn->w_out[j+1][k] * delta_out[k];
                    delta_hid[j] = sum * dReLU(cnn->in_h[j]);
                }

                /* akumulacija gradijenata WOut */
                for (int k = 0; k < cnn->K; k++) {
                    grad_w_out[0][k] += delta_out[k];
                    for (int j = 0; j < cnn->J; j++)
                        grad_w_out[j+1][k] += delta_out[k] * cnn->a[j];
                }

                /* akumulacija gradijenata WIn */
                for (int j = 0; j < cnn->J; j++) {
                    grad_w_in[0][j] += delta_hid[j];
                    for (int p = 0; p < cnn->n; p++)
                        grad_w_in[p+1][j] += delta_hid[j] * cnn->x[p];
                }
            }

            /* update tezina */
            for (int k = 0; k < cnn->K; k++) {
                cnn->w_out[0][k] -= cnn->alpha * grad_w_out[0][k] / currentBatch;
                for (int j = 0; j < cnn->J; j++)
                    cnn->w_out[j+1][k] -= cnn->alpha * grad_w_out[j+1][k] / currentBatch;
            }

            for (int j = 0; j < cnn->J; j++) {
                cnn->w_in[0][j] -= cnn->alpha * grad_w_in[0][j] / currentBatch;
                for (int p = 0; p < cnn->n; p++)
                    cnn->w_in[p+1][j] -= cnn->alpha * grad_w_in[p+1][j] / currentBatch;
            }

            for (int p = 0; p < cnn->n + 1; p++) free(grad_w_in[p]);
            free(grad_w_in);
            for (int j = 0; j < cnn->J + 1; j++) free(grad_w_out[j]);
            free(grad_w_out);
        }

        loss /= trainSize;
        printf("Epoch %d finished, loss = %.4f\n", epoch + 1, loss);
    }

    free(indices);
    for (int i = 0; i < trainSize; i++) free(y[i]);
    free(y);
    free(delta_out);
    free(delta_hid);
}



int pogodi(CNN *cnn, IMAGE image)
{
    for (int p = 0; p < cnn->n; p++)
    {
        cnn->x[p] = image.data[p];
    }

    forward(cnn);

    int idx = 0;
    double max = cnn->b[0];
    for (int k = 1; k < cnn->K; k++)
    {
        if (cnn->b[k] > max)
        {
            max = cnn->b[k];
            idx = k;
        }
    }

    return idx;
}

float test(CNN *cnn, IMAGE *images)
{
    int correct = 0;
    for (int m = 0; m < testSize; m++)
    {
        int guess = pogodi(cnn, images[m]);
        if (guess == images[m].label)
        {
            correct++;
        }
    }
    return (float)correct / (float)testSize;
}
