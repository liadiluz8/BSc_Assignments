#include <fcntl.h>      /* open */
#include <unistd.h>     /* exit */
#include <sys/ioctl.h>  /* ioctl */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "message_slot.h"

int main(int argc, char *argv[]){
    char *msg;
    unsigned int channel_id;
    int fd;

    if (argc != 4){
        perror("Invalid number of arguments");
        exit(1);
    }
    
    channel_id = atoi(argv[2]);
    msg = argv[3];

    // Flow
    // 1. Open device file
    fd = open(argv[1], O_RDWR);
    if (fd < 0){
        // Error in opening the file
        fprintf(stderr, "Can't open device file: %s\n", argv[1]);
        exit(1);
    }
    // 2.Set channel id
    if (ioctl(fd, MSG_SLOT_CHANNEL, channel_id) != 0){
		fprintf(stderr, "ioctl failed. Error number: %s\n", strerror(errno));
		close(fd);
        exit(1);
    }
    // 3. Write message to message slot file
    if (write(fd, msg, strlen(msg)) <= 0){
        fprintf(stderr, "write failed. Error number: %s\n", strerror(errno));
		close(fd);
        exit(1);
    }
    // 4. Close device
    close(fd);
    printf("Successfuly sent\n");
    // 5. Exit
    exit(0);
}
