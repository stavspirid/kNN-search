#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cblas.h>
#include <unistd.h>
#include "knn.h"

// #define RAND_MAX 1024

typedef struct {
    double distance;
    int index;
} Neighbor;

void swap(Neighbor *a, Neighbor *b) {
    Neighbor temp = *a;
    *a = *b;
    *b = temp;
}

int partition(Neighbor *set, int low, int high) {
    double pivot = set[high].distance;
    int i = low - 1;
    
    for (int j = low; j < high; j++) {
        if (set[j].distance <= pivot) {
            swap(&set[++i], &set[j]);
        }
    }
    swap(&set[i + 1], &set[high]);
    return i + 1;
}

void quickselect(Neighbor *set, int low, int high, int k) {
    if (low < high) {
        int pi = partition(set, low, high);
        
        if (pi > k - 1) {
            quickselect(set, low, pi - 1, k);
        } else if (pi < k - 1) {
            quickselect(set, pi + 1, high, k);
        }
    }
    
    // Sort first k elements
    for (int i = 0; i < k - 1; i++) {
        for (int j = i + 1; j < k; j++) {
            if (set[i].distance > set[j].distance) {
                swap(&set[i], &set[j]);
            }
        }
    }
}

void blockQuickselect(Neighbor *blockDistances, int cSize, int curBlockSize, int neighbors) {
    // For each query in the block, find its k nearest neighbors
    for (int q = 0; q < curBlockSize; q++) {
        Neighbor *queryDistances = &blockDistances[q * cSize];
        quickselect(queryDistances, 0, cSize - 1, neighbors);
    }
}

double *createSet(int setSize, int dim) {
    return (double *)malloc(setSize * dim* sizeof(double));
}

void fillRandomData(double *set, int setSize, int dim) {
    for (int i = 0; i < setSize * dim; i++) {
        set[i] = ((double)rand() / RAND_MAX) * 100.0; // Random values 0-100
    }
}

void calcSquare(double *set, int setSize, int dim, double *squared) {
    printf("Calculating squared values for set of size %d and dimension %d\n", setSize, dim);
    sleep(1); // Sleep for 1 second to allow time for printing
    for (int i = 0; i < setSize; i++) {
        squared[i] = 0.0;
        for (int j = 0; j < dim; j++) {
            squared[i] += set[i * dim + j] * set[i * dim + j];
        }
    }
}

void calcDistances(double *C, double *qBlock, double *cSquared, double *D, 
                   int cSize, int qBlockSize, int dim) {
    
    // Step 1: Compute 2CQ^T using OpenBLAS (cSize x qBlockSize)
    double *cqBlock = (double *)malloc(cSize * qBlockSize * sizeof(double));
    // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
    //             cSize, qBlockSize, dim, 2.0, C, cSize, qBlock, qBlockSize, 0.0, cqBlock, cSize);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                cSize, qBlockSize, dim, 2.0, C, dim, qBlock, dim, 0.0, cqBlock, qBlockSize); // was cSize instead of qBlockSize before


    // Step 2: Compute Q^2 element-wise (qBlockSize x 1)
    double *qBlockSquared = (double *)malloc(qBlockSize * sizeof(double));
    calcSquare(qBlock, qBlockSize, dim, qBlockSquared);
    
    // Step 3: Compute distances: D = sqrt(C^2 + Q^2 - 2CQ^T)
    for (int i = 0; i < cSize; i++) {
        for (int j = 0; j < qBlockSize; j++) {
            D[i * qBlockSize + j] = sqrt(cSquared[i] + qBlockSquared[j] - 
                                        cqBlock[i * qBlockSize + j]);
        }
    }

    free(cqBlock);
    free(qBlockSquared);
}

void blockedKNNsearch(double *C, double *Q, int cSize, int qSize, int dim, int neighbors, int numberOfBlocks) {
    printf("In blockedKNNsearch with cSize: %d, qSize: %d, dim: %d, neighbors: %d, numberOfBlocks: %d\n\n", cSize, qSize, dim, neighbors, numberOfBlocks);

    sleep(2); 
    int blockSize = qSize / numberOfBlocks; // Size of each block
    int curBlockSize;  // Size of the current block

    // Compute C^2 element-wise (cSize x 1)
    double *cSquared = createSet(cSize, 1);
    calcSquare(C, cSize, dim, cSquared);

    // Allocate memory for nearest neighbors and fill with initial values
    Neighbor *nearestNeighbors = (Neighbor *)malloc(qSize * neighbors * sizeof(Neighbor));
    for (int i = 0; i < qSize * neighbors; i++) {
        nearestNeighbors[i].distance = INFINITY;
        nearestNeighbors[i].index = -1;
    }

    for (int block = 0; block < numberOfBlocks; block++) {
        // Size of the current block
        if (block == numberOfBlocks - 1) {
            curBlockSize = qSize - block * blockSize; // Last block may be larger
        } else {
            curBlockSize = blockSize;
        }

        // Allocate Blocked Distances matrix
        double *D = (double *)malloc(curBlockSize * cSize * sizeof(double));

        calcDistances(C, &Q[block * blockSize * dim], cSquared, 
                      D, cSize, curBlockSize, dim);
        
        printf("Processing block %d/%d with %d query points\n", 
                block + 1, numberOfBlocks, curBlockSize);


        // Convert the entire distance block to Neighbor format
        // Layout: [Q0_C0, Q0_C1, ..., Q0_C(cSize-1), Q1_C0, Q1_C1, ..., Q1_C(cSize-1), ...]
        Neighbor *blockDistances = (Neighbor *)malloc(curBlockSize * cSize * sizeof(Neighbor));
        
        for (int q = 0; q < curBlockSize; q++) {
            for (int c = 0; c < cSize; c++) {
                blockDistances[q * cSize + c].distance = D[c * curBlockSize + q];
                blockDistances[q * cSize + c].index = c; // Original index in C
            }
        }
        
        // for (int q = 0; q < curBlockSize; q++) {
        //     printf("Row %d: ", q);
        //     for (int c = 0; c < cSize; c++) {
        //         printf("C[%d]=%.3f ", blockDistances[q * cSize + c].index, 
        //                blockDistances[q * cSize + c].distance);
        //     }
        //     printf("\n");
        // }

        blockQuickselect(blockDistances, cSize, curBlockSize, neighbors);

        // for (int q = 0; q < curBlockSize; q++) {
        //     printf("Row %d: ", q);
        //     for (int c = 0; c < cSize; c++) {
        //         printf("C[%d]=%.3f ", blockDistances[q * cSize + c].index, 
        //                blockDistances[q * cSize + c].distance);
        //     }
        //     printf("\n");
        // }

        for (int q = 0; q < curBlockSize; q++) {
            for (int c = 0; c < neighbors; c++) {
                int nearestIndex = blockDistances[q * cSize + c].index;
                nearestNeighbors[q * neighbors + c].distance = blockDistances[q * cSize + c].distance;
                nearestNeighbors[q * neighbors + c].index = nearestIndex;
            }
        }
        for (int q = 0; q < curBlockSize; q++) {
            printf("Query %d: ", q);
            for (int c = 0; c < neighbors; c++) {
                printf("C[%d]=%.3f ", nearestNeighbors[q * neighbors + c].index, 
                       nearestNeighbors[q * neighbors + c].distance);
            }
            printf("\n");
        }

        // Free allocated memory
        free(blockDistances);
        free(D);
    }

    // Free allocated memory
    free(cSquared);
    free(nearestNeighbors);
}

int main(void) {
    double *C, *Q, *D;
    int cSize = 5000;
    int qSize = 5000;
    int dim = 14;
    int neighbors = 3;
    int numberOfBlocks = 300;

    // C: Known set
    C = createSet(cSize, dim);
    // Q: Query set
    Q = createSet(qSize, dim);
    // D: Distances of nearest neighbors
    D = createSet(cSize, dim);
    // nearestN: array of distances and indidces
    Neighbor *nearestN = (Neighbor *)malloc(qSize * neighbors * sizeof(Neighbor));

    fillRandomData(C, cSize, dim);
    fillRandomData(Q, qSize, dim);

    // for (int i = 0; i < cSize; i++) {
    //     printf("C[%d]: ", i);
    //     for (int j = 0; j < dim; j++) {
    //         printf("%f ", C[i * dim + j]);
    //     }
    //     printf("\n");
    // }
    // for (int i = 0; i < qSize; i++) {
    //     printf("Q[%d]: ", i);
    //     for (int j = 0; j < dim; j++) {
    //         printf("%f ", Q[i * dim + j]);
    //     }
    //     printf("\n");
    // }

    sleep(1); // Sleep for 1 second to allow time for printing



    blockedKNNsearch(C, Q, cSize, qSize, dim, neighbors, numberOfBlocks);

    free(C);
    free(Q);
    free(D);
    free(nearestN);
    printf("Finished KNN search.\n");
    
    return 0;
}