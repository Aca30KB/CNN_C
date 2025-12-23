#ifndef DATA_H
#define DATA_H

#include "image.h"

typedef struct _data
{
    const char *filename; /* naziv fajla sa podacima */
    int nbRecords;        /* broj slogova (redova) */
} DATA;

/* ucitavanje podataka iz CSV fajla u niz IMAGE struktura */
void loadData(DATA *data, IMAGE *images);

#endif /* DATA_H */
