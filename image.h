#ifndef IMAGE_H
#define IMAGE_H

#ifndef EXTERN
#define EXTERN extern
#endif

EXTERN int testSize;
EXTERN int trainSize;

#include <unistd.h>

/* pravi se struktira slike i labela za svaki rekord ponaosob, a kasnije se svi
rekordi tj. redovi ulaznih fajlova smestaju u po jedan element niza tipa IMAGE koji ima kapacitet
da sakupi sve primere iz test fajla, tj. iz trening fajla */
typedef struct _image
{
    int n;        /* duzina slike */
    double *data; /* podaci (pikseli) */
    int label;    /* kontrolna vrednost */
} IMAGE;

/* funkcija alokacije memorije za podatke */
void allocData(IMAGE *, int);
/* funkcija oslobadjanja memorije podataka */
void freeData(IMAGE **, int);
#endif /* IMAGE_H */