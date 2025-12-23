#include "cnn.h"
#include <assert.h>
#include <math.h>
#include <gsl/gsl_math.h>

double leak = 0.01;
double sigma = 1.0;

void allocCNN(CNN *cnn, int size)
{
    int l;
    int j;
    int n;
    int k;
    cnn->L = 5;
    cnn->alpha = 0.01;
    cnn->rows = 28;
    cnn->cols = 28;
    cnn->J = 32;
    cnn->K = 10;
    cnn->n = cnn->rows * cnn->cols;

    /* x ulazni sloj */

    cnn->x = (double *)calloc((size_t)cnn->n, sizeof(double));
    assert(cnn->x != NULL);

    /* ulazne vrednosti u cvorove skrivenih slojeva */

    cnn->in = (double **)calloc((size_t)(cnn->L + 1), sizeof(double *));
    assert(cnn->in != NULL);
    for (l = 0; l < cnn->L + 1; l++)
    {
        if (l == cnn->L)
        {
            cnn->in[l] = (double *)calloc((size_t)cnn->K, sizeof(double));
            assert(cnn->in[l] != NULL);
        }
        else
        {
            cnn->in[l] = (double *)calloc((size_t)cnn->J, sizeof(double));
            assert(cnn->in[l] != NULL);
        }
    }
    /* a  skriveni slojevi */

    cnn->a = (double **)calloc((size_t)cnn->L, sizeof(double *));
    assert(cnn->a != NULL);
    for (l = 0; l < cnn->L; l++)
    {
        cnn->a[l] = (double *)calloc((size_t)cnn->J, sizeof(double));
        assert(cnn->a[l] != NULL);
    }

    /* izlazni sloj */

    cnn->b = (double *)calloc((size_t)cnn->K, sizeof(double));
    assert(cnn->b != NULL);

    /* w_in (matrica tezina izmedju ulaznog sloja prvog skrivenog sloja) */

    cnn->w_in = (double **)calloc((size_t)cnn->n + 1, sizeof(double *));
    assert(cnn->w_in != NULL);
    for (n = 0; n < cnn->n + 1; n++)
    {
        cnn->w_in[n] = (double *)calloc((size_t)cnn->J, sizeof(double));
        assert(cnn->w_in[n] != NULL);
    }

    cnn->w_in_T = (double **)calloc((size_t)cnn->J, sizeof(double *));
    assert(cnn->w_in_T != NULL);
    for (j = 0; j < cnn->J; j++)
    {
        cnn->w_in_T[j] = (double *)calloc((size_t)cnn->n + 1, sizeof(double));
        assert(cnn->w_in_T[j] != NULL);
    }

    /* w (tezine) */
    cnn->w = (double ***)calloc((size_t)(cnn->L), sizeof(double **));
    assert(cnn->w != NULL);
    for (l = 0; l < cnn->L + 1; l++)
    {
        cnn->w[l] = (double **)calloc((size_t)(cnn->J + 1), sizeof(double *));
        assert(cnn->w[l] != NULL);
        for (j = 0; j < cnn->J + 1; j++)
        {
            cnn->w[l][j] = (double *)calloc((size_t)cnn->J, sizeof(double));
            assert(cnn->w[l][j] != NULL);
        }
    }

    /* w_out (matrica tezina izmedju skrivenog slija i izlaznog sloja) */

    cnn->w_out = (double **)calloc((size_t)(cnn->J + 1), sizeof(double *));
    assert(cnn->w_out != NULL);
    for (j = 0; j < cnn->J + 1; j++)
    {
        cnn->w_out[j] = (double *)calloc((size_t)cnn->K, sizeof(double));
        assert(cnn->w_out[j] != NULL);
    }

    cnn->w_out_T = (double **)calloc((size_t)cnn->K, sizeof(double *));
    for (k = 0; k < cnn->K; k++)
    {
        cnn->w_out_T[k] = (double *)calloc((size_t)(cnn->J + 1), sizeof(double));
    }
}

void freeCNN(CNN *cnn)
{
    int l;
    int n;
    int j;
    free(cnn->x);
    for (l = 0; l < cnn->L + 1; l++)
    {
        free(cnn->in[l]);
        free(cnn->a[l]);
        for (j = 0; j < cnn->J + 1; j++)
        {
            free(cnn->w[l][j]);
        }
    }
    free(cnn->in);
    free(cnn->a);
    free(cnn->w);
    for (n = 0; n < cnn->n + 1; n++)
    {
        free(cnn->w_in[n]);
    }
    free(cnn->w_in);
    for (j = 0; j < cnn->J + 1; j++)
    {
        free(cnn->w_in_T[j]);
    }
    for (j = 0; j < trainSize; j++)
    {
        free(cnn->y[j]);
    }
    free(cnn->y);
    free(cnn->w_in_T);
}

void transpose(double **a, double **t, int M, int N)
{
    int m;
    int n;

    for (m = 0; m < M; m++)
    {
        for (n = 0; n < N; n++)
        {
            t[n][m] = a[m][n];
        }
    }
}

void initweights(CNN *cnn)
{
    int l;  /* brojac slojeva */
    int k;  /* brojac izlaznog sloja */
    int j1; /* brojac cvorova u sjrivenim slojevima */
    int j2; /* brojac cvorova u sjrivenim slojevima */
    int n;  /* brojac cvorova u ulaznom sloju */

    /* ulazni sloj */

    for (n = 0; n < cnn->n + 1; n++)
    {
        for (j1 = 0; j1 < cnn->J; j1++)
        {
            cnn->w_in[n][j1] = gsl_ran_gaussian(r, sigma) * sqrt(1.0 / 784.0);
        }
    }

    /* skriveni slojevi */

    for (l = 0; l < cnn->L; l++)
    {
        for (j1 = 0; j1 < cnn->J + 1; j1++)
        {
            for (j2 = 0; j2 < cnn->J; j2++)
            {
                cnn->w[l][j1][j2] = gsl_ran_gaussian(r, sigma) * sqrt(1.0 / 32.0);
            }
        }
    }

    /* izlazni sloj */

    for (j1 = 0; j1 < cnn->J + 1; j1++)
    {
        for (k = 0; k < cnn->K; k++)
        {
            cnn->w_out[j1][k] = gsl_ran_gaussian(r, sigma) * sqrt(1.0 / 32.0);
        }
    }
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double dSigmoid(double x)
{
    return exp(-x) / pow((1.0 + exp(-x)), 2.0);
}

double ReLU(double x)
{
    return GSL_MAX(x, 0.0);
}

double dReLU(double x)
{
    if (x <= 0.0)
    {
        return leak;
    }
    return 1.0;
}

double softmaxsum(double *b)
{
    double sum = 0.0;
    int m;
    for (m = 0; m < 10; m++)
    {
        sum += exp(b[m]);
    }
    return sum;
}

int maxindex(double *y)
{
    int index;
    int m;
    double max;
    index = 0;
    max = y[0];
    for (m = 1; m < 10; m++)
    {
        if (max < y[m])
        {
            max = y[m];
            index = m;
        }
    }
    return index;
}

void softmax(double *b, double *y)
{
    int m;
    double sum;
    sum = softmaxsum(b);
    for (m = 0; m < 10; m++)
    {
        y[m] = exp(b[m]) / sum;
    }
}

double dot(double *w, double *a, int size)
{
    int i;
    double sum;
    sum = 0.0;
    for (i = 0; i < size; i++)
    {
        sum += w[i + 1] * a[i];
    }
    sum += w[0];
    return sum;
}

void onehot(CNN *cnn, IMAGE *image, int size)
{
    int i;
    int j;

    for (i = 0; i < size; i++)
    {
        for (j = 0; j < cnn->J; j++)
        {
            if (j == image[i].label)
            {
                cnn->y[i][j] = 1.0;
            }
            else
            {
                cnn->y[i][j] = 0.0;
            }
        }
    }
}

void backPropLearning(CNN *cnn, IMAGE *train_images)
{
    int epoch; /* brojac epoha */
    int n;
    int i;          /* brojac trening slija */
    int k;          /* brojac u izlaznom sloju */
    int j;          /* brojac cvorova u slojevima */
    int p;          /* brojac pijsela */
    int l;          /* brojac slojeva */
    int j1;         /* prvi brojac u matrici tezina */
    int j2;         /* drugi brojac u matrici tezina */
    int *indeks;    /* indeks trening podataka za permutovanje */
    double **delta; /* vektor gresaka */
    double *y_max;  /* normalizovan izlazni vektor */
    double **y;     /* onehot matrica */

    /* delta */
    delta = (double **)calloc((size_t)(cnn->L + 1), sizeof(double *));
    assert(delta != NULL);
    for (l = 0; l < cnn->L + 1; l++)
    {
        delta[l] = (double *)calloc((size_t)cnn->J, sizeof(double));
        assert(delta[l] != NULL);
    }

    y_max = (double *)calloc((size_t)cnn->J, sizeof(double));
    assert(y_max != NULL);

    indeks = (int *)calloc((size_t)trainSize, sizeof(int));
    assert(indeks != NULL);
    for (i = 0; i < trainSize; i++)
    {
        indeks[i] = i;
    }
    /* onehot matrica */
    y = (double **)calloc((size_t)size, sizeof(double *));
    assert(cnn->y != NULL);
    for (j = 0; j < size; j++)
    {
        y[j] = (double *)calloc((size_t)cnn->J, sizeof(double));
        assert(cnn->y[j] != NULL);
    }
    /*onehot matrica */

    onehot(cnn, train_images, trainSize);
    /* inicijalizacija matrica tezina */
    initweights(cnn);
    epoch = 0;
    do
    {
        /* muckanje indeksa trening seta */
        gsl_ran_shuffle(r, indeks, (size_t)trainSize, sizeof(int));

        for (i = 0; i < trainSize; i++)
        {
            /* inicijalizacija pocetnog sloja */
            for (p = 0; p < train_images[0].n; p++)
            {
                cnn->x[p] = train_images[indeks[i]].data[p] / 255.0;
            }

            /* prostiranje unapred da bi se izracunali izlazi */

            transpose(cnn->w_in, cnn->w_in_T, cnn->n, cnn->J);

            /* veza izmedju ulaznog i prvog skrivenog sloja */
            for (j = 0; j < cnn->J; j++)
            {
                cnn->in[0][j] = dot(cnn->w_in_T[j], cnn->x, cnn->n);
                cnn->a[0][j] = sigmoid(cnn->in[0][j]);
            }

            transpose(cnn->w_in_T, cnn->w_in, cnn->J, cnn->n);

            transpose(cnn->w_out, cnn->w_out_T, cnn->J, cnn->K);

            for (l = 1; l < cnn->L + 1; l++)
            {
                if (l == cnn->L)
                {
                    for (k = 0; k < cnn->K; k++)
                    {
                        cnn->in[l][k] = dot(cnn->w_out_T[k], cnn->a[l - 1], cnn->J);
                        cnn->b[k] = sigmoid(cnn->in[l][k]);
                    }
                }
                else
                {
                    for (j = 0; j < cnn->J; j++)
                    {
                        cnn->in[l][j] = dot(cnn->w[l - 1][j], cnn->a[l - 1], cnn->J);
                        cnn->a[l][j] = sigmoid(cnn->in[l][j]);
                    }
                }
            }

            transpose(cnn->w_out_T, cnn->w_out, cnn->K, cnn->J);

            /* prostiranje unazad od izlaznog sloja ja ulaznom */

            /* najpre ulazni sloj normalizujemo */

            softmax(cnn->b, y_max);

            for (j = 0; j < cnn->J; j++)
            {
                delta[cnn->L][j] = dSigmoid(cnn->in[cnn->L][j]) * (cnn->y[indeks[i]][j] - y_max[j]);
            }

            for (l = cnn->L - 1; l >= 0; l--)
            {
                for (j = 0; j < cnn->J; j++)
                {
                    delta[l][j] = dSigmoid(cnn->in[l][j]) * dot(cnn->w[l][j], delta[l], cnn->J);
                }
            }

            /* azuriranje svake tezine koriscenjem delti */

            for (l = 0; l < cnn->L; l++)
            {
                for (j1 = 0; j1 < cnn->J; j1++)
                {
                    for (j2 = 0; j2 < cnn->J; j2++)
                    {
                        cnn->w[l][j1][j2] = cnn->w[l][j1][j2] + cnn->alpha * cnn->a[l][j1] * delta[l][j2];
                    }
                }
            }

            for (n = 0; n < cnn->n; n++)
            {
                for (j = 0; j < cnn->J; j++)
                {
                    cnn->w_in[n][j] = cnn->w_in[n][j] + cnn->alpha * cnn->a[0][j] * delta[0][j];
                }
            }

            for (j = 0; j < cnn->J + 1; j++)
            {
                for (k = 0; k < cnn->K; k++)
                {
                    cnn->w_out[j][k] = cnn->w_out[j][k] + cnn->alpha * cnn->b[k] * delta[cnn->L][j];
                }
            }
        }
        fprintf(stdout, "%d\n", epoch);
    } while (epoch++ < epochs);
    free(delta);
    free(indeks);
    free(y_max);
}

int pogodi(CNN *cnn, IMAGE image)
{
    int p;
    int j;
    int l;
    int k;
    for (p = 0; p < cnn->n; p++)
    {
        cnn->x[p] = image.data[p] / 255.0;
    }
    transpose(cnn->w_in, cnn->w_in_T, cnn->n, cnn->J);

    for (j = 0; j < cnn->J; j++)
    {
        cnn->in[0][j] = dot(cnn->w_in_T[j], cnn->x, cnn->n);
        cnn->a[0][j] = sigmoid(cnn->in[0][j]);
    }

    for (l = 1; l < cnn->L; l++)
    {
        for (j = 0; j < cnn->J; j++)
        {
            cnn->in[l][j] = dot(cnn->w[l][j], cnn->a[l], cnn->J);
            cnn->a[l][j] = sigmoid(cnn->in[l][j]);
        }
    }

    for (k = 0; k < cnn->K; k++)
    {
        cnn->in[cnn->L][k] = dot(cnn->w_out[k], cnn->a[cnn->L - 1], cnn->K);
        cnn->b[k] = sigmoid(cnn->in[cnn->L][k]);
    }

    return maxindex(cnn->b);
}

float test(CNN *cnn, IMAGE *image)
{
    int guess;
    int correct;
    int m;
    correct = 0;

    for (m = 0; m < testSize; m++)
    {
        guess = pogodi(cnn, image[m]);
        if (guess == image[m].label)
        {
            correct++;
        }
    }
    return (float)correct / (float)testSize;
}