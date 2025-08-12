#ifndef KNN
#define KNN

typedef struct {
    double distance;
    int index;
} Neighbor;

void swap(Neighbor *a, Neighbor *b);

int partition(Neighbor *set, int low, int high);

void quickselect(Neighbor *set, int low, int high, int k);

void blockQuickselect(Neighbor *blockDistances, int cSize, int curBlockSize, int neighbors);

double *createSet(int setSize, int dim);

void fillRandomData(double *set, int setSize, int dim);

void calcSquare(double *set, int setSize, int dim, double *squared);

void calcDistances(double *C, double *qBlock, double *cSquared, double *D, int cSize, int qBlockSize, int dim);

void blockedKNNsearch(double *C, double *Q, int cSize, int qSize, int dim, int neighbors, int numberOfBlocks)

// Find the k nearest neighbors for each query in the block
void blockQuickselect(Neighbor *blockDistances, int cSize, int curBlockSize, int neighbors);

#endif // KNN   