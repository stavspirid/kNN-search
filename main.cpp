#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

u_int8_t mails = 0;

void* routine(void* arg) {
    printf("Hello from threads\n");
    for (int i = 0; i < 100; i++) {
        mails++;
    }
    sleep(2);
    printf("Ending thread\n");

    return NULL;
}

int main(int argc, char* argv[]) {
    pthread_t p1, p2;
    if (pthread_create(&p1, NULL, &routine, NULL) != 0) {
        return 1;
    }
    if (pthread_create(&p2, NULL, &routine, NULL) != 0) {
        return 2;
    }
    if (pthread_join(p1, NULL) != 0) {
        return 3;
    }
    if (pthread_join(p2, NULL) != 0) {
        return 4;
    }

    printf("Number of mails: %d\n", mails);

    return 0;
}