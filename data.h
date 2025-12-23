#ifndef DATA_H
#define DATA_H
#include "image.h"

typedef struct _data
{
    const char *filename;
    int nbRecords;
} DATA;

/* funkcija za ucitavanje podataka */
void loadData(DATA *, IMAGE *);

#endif /* DATA_H */