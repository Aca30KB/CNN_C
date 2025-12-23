#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

void loadData(DATA *data, IMAGE *image)
{
    int i; /* brojac piksela */
    int m;

    /* provera postojanja fajla i pokusaj njegovog otvaranja */

    if (access(data->filename, F_OK) == 0)
    {
        FILE *in;
        char buffer[3200]; /* bufer dovoljne velicine da se smesti jedna linija fajla*/
        char *podatak;
        in = fopen(data->filename, "r");
        if (in == NULL)
        {
            printf("Vrednost errno: %d", errno);
            printf("\nPoruka greske %s.", strerror(errno));
            perror("Poruka funkcije perror");
            exit(EXIT_FAILURE);
        }

        m = 0;
        while (fgets(buffer, sizeof(buffer), in))
        {
            podatak = strtok(buffer, ",");

            image[m].label = atoi(podatak);

            for (i = 0; i < image[m].n; i++)
            {
                podatak = strtok(NULL, ",");
                image[m].data[i] = atof(podatak) / 255.0;
            }
            m++;
        }

        /* zatvaranje fajla */
        if (fclose(in) != 0)
        {
            perror("Greska pri zatvaranju fajla");
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        fprintf(stderr, "Fajl %s ne postoji!\n", data->filename);
        exit(EXIT_FAILURE);
    }
}