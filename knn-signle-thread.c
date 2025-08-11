#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cblas.h>
#include "knn.h"

#define RAND_MAX 1024

// void swap(Neighbor *n, int a, int b) {
//     Neighbor temp = n[a];
//     n[a] = n[b];
//     n[b] = temp;
// }   

// int partition(Neighbor *set , int low, int high) {
//     double pivot = set[high].distance;
//     int i = low - 1;
    
//     for (int j = low; j < high; j++) {
//         if (set[j].distance  < pivot) {
//             i++;
//             swap(set, i, j);
//         }
//     }
    
//     swap(set, i+1, high);
//     return i + 1;
// }


// int quickSelect(Neighbor* set, int low, int high, int neighbors) {
//     if (low < high) {
//         int pivot = partition(set, low, high);
        
//         if (pivot == neighbors) {
//             return set[pivot];
//         } else if (pivot > neighbors) {
//             return quickSelect(set, low, pivot - 1, neighbors);
//         } else {
//             return quickSelect(set, pivot + 1, high, neighbors);
//         }
//     }
//     return -1;
// }

// ====================== Quickselect Implementation ======================
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
// ======================

void createSet(double *set, int setSize, int dim) {
    return (double *)malloc(setSize * dim* sizeof(double));
}

void fillRandomData(double *set, int setSize, int dim) {
    for (int i = 0; i < setSize * dim; i++) {
        set[i] = ((double)rand() / RAND_MAX) * 100.0; // Random values 0-100
    }
}

void calcSquare(double *set, int setSize, int dim, double *squared) {
    for (int i = 0; i < setSize; i++) {
        for (int j = 0; j < dim; j++) {
            squared[i] += set[i * dim + j] * set[i * dim + j];
        }
    }
}

void calcDistances(double *C, double *qBlock, double *cSquared, double *D, 
                   int cSize, int qBlockSize, int dim) {
    
    // Step 1: Compute 2CQ^T using OpenBLAS (cSize x qBlockSize)
    double *cqBlock = (double *)malloc(cSize * qBlockSize * sizeof(double));
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                cSize, qBlockSize, dim, 2.0, C, cSize, qBlock, qBlockSize, 0.0, cqBlock, cSize);
    
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
    int blockSize = qSize / numberOfBlocks; // Size of each block
    int curBlockSize;  // Size of the current block

    // Compute C^2 element-wise (cSize x 1)
    double *cSquared = (double *)malloc(cSize * sizeof(double));  
    calcSquare(C, cSize, dim, cSquared);

    // Allocate memory for nearest neighbors and fill with initial values
    Neighbor *nearestNeighbors = (Neighbor *)malloc(qSize * neighbors * sizeof(Neighbor));
    for (int i = 0; i < qSize * neighbors; i++) {
        nearestNeighbors[i].distance = INFINITY;
        nearestNeighbors[i].index = -1;
    }

    for (int block = 0; block < numberOfBlocks; block++) {
        // Allocate Blocked Distances matrix
        double *D = (double *)malloc(cSize * curBlockSize * sizeof(double));

        // Size of the current block
        if (block == numberOfBlocks - 1) {
            curBlockSize = qSize - block * blockSize; // Last block may be smaller
        } else {
            curBlockSize = blockSize;
        }

        calcDistances(C, &Q[block * blockSize * dim], cSquared, 
                      &D[block * blockSize * cSize], cSize, curBlockSize, dim);
        
        printf("Processing block %d/%d with %d query points\n", 
               block + 1, numberOfBlocks, curBlockSize);

        quickselect(D, 0, cSize * curBlockSize - 1, neighbors - 1);

        // Free allocated memory
        free(D);
    }

    // Free allocated memory
    free(cSquared);
}

int main(void) {
    double *C, *Q, *D;
    int cSize = 2000;
    int qSize = 2000;
    int dim = 2;
    int neighbors = 3;
    int numberOfBlocks = 10;

    // C: Known set
    createSet(C, cSize, dim);
    // Q: Query set
    createSet(Q, qSize, dim);
    // D: Distances of nearest neighbors
    createSet(D, cSize, dim);
    // nearestN: array of distances and indidces
    Neighbor *nearestN = (Neighbor *)malloc(qSize * neighbors * sizeof(Neighbor));

    for (int i = 0; i < qSize; i++) {
        printf("Query %d:\n", i);
        for(int j = 0; j < neighbors; j++) {
            printf("Number: %d  Distance: %d  Index: %d\n",j, nearestN[i * neighbors + j].distance, nearestN[i * neighbors + j].index);
        }
    }
    
    return 0;
}