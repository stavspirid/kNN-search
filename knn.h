#ifndef KNN
#define KNN

typedef struct {
    double distance;
    int index;
} Neighbor;

void createSet(double *set, int setSize, int dim);

void calcSquare(double *set, int setSize, int dim, double *squared);

void calcDistances(double *C, double *qBlock, double *cSquared, double *D, int cSize, int qBlockSize, int dim);

// void blockedKNNsearch(double *C, double *Q, int cSize, int qSize, int dim, int neighbors);

#endif // KNN   