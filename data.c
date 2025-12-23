#include "data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

void loadData(DATA *data, IMAGE *images)
{
    FILE *in = fopen(data->filename, "r");
    if (in == NULL)
    {
        fprintf(stderr, "Greska pri otvaranju fajla %s\n", data->filename);
        fprintf(stderr, "errno = %d: %s\n", errno, strerror(errno));
        exit(EXIT_FAILURE);
    }

    char buffer[4096];
    int m = 0;

    while (fgets(buffer, sizeof(buffer), in))
    {
        if (m >= data->nbRecords)
        {
            fprintf(stderr, "Greska: fajl %s ima vise redova nego sto je ocekivano (%d)\n",
                    data->filename, data->nbRecords);
            exit(EXIT_FAILURE);
        }

        if (buffer[0] == '\n' || buffer[0] == '\0')
            continue;

        char *token = strtok(buffer, ",");
        if (token == NULL)
        {
            fprintf(stderr, "Greska: nevalidna linija u fajlu %s (red %d)\n",
                    data->filename, m);
            exit(EXIT_FAILURE);
        }

        images[m].label = atoi(token);

        for (int i = 0; i < images[m].n; i++)
        {
            token = strtok(NULL, ",");
            if (token == NULL)
            {
                fprintf(stderr, "Greska: nedovoljno podataka u liniji %d fajla %s\n",
                        m, data->filename);
                exit(EXIT_FAILURE);
            }
            images[m].data[i] = atof(token) / 255.0;
        }

        m++;
    }

    if (m < data->nbRecords)
    {
        fprintf(stderr, "Upozorenje: fajl %s ima samo %d redova, a ocekivano je %d\n",
                data->filename, m, data->nbRecords);
    }

    if (fclose(in) != 0)
    {
        perror("Greska pri zatvaranju fajla");
        exit(EXIT_FAILURE);
    }
}

