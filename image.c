#include "image.h"
#include <stdlib.h>
#include <assert.h>

void allocData(IMAGE *images, int size)
{
    for (int m = 0; m < size; m++)
    {
        images[m].n = 784;
        images[m].data = (double *)calloc((size_t)images[m].n, sizeof(double));
        assert(images[m].data != NULL);
    }
}

void freeData(IMAGE *images, int size)
{
    for (int m = 0; m < size; m++)
    {
        free(images[m].data);
        images[m].data = NULL;
        images[m].n = 0;
    }
}
