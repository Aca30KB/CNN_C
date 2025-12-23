#ifndef IMAGE_H
#define IMAGE_H

typedef struct _image
{
    int label;    /* oznaka cifre 0–9 */
    int n;        /* broj piksela (784) */
    double *data; /* niz piksela normalizovanih na [0, 1] */
} IMAGE;

/* alokacija polja data za niz slika */
void allocData(IMAGE *images, int size);

/* oslobadjanje polja data za niz slika */
void freeData(IMAGE *images, int size);

#endif /* IMAGE_H */
