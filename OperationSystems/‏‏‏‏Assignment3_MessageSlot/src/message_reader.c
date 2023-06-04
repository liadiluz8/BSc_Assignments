#include <fcntl.h>      /* open */
#include <unistd.h>     /* exit */
#include <sys/ioctl.h>  /* ioctl */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "message_slot.h"

int main(int argc, char *argv[]){
    unsigned int channel_id;
    char msg[BUF_LEN];
    int fd, b_num = 0;

    if (argc != 3){
        perror("Invalid number of arguments");
        exit(1);
    }
    
    channel_id = atoi(argv[2]);

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
    // 3. Read message from message slot file to msg
    b_num = 0;
    if ((b_num = read(fd, msg, BUF_LEN)) <= 0){
        fprintf(stderr, "read failed. Error number: %s\n", strerror(errno));
		close(fd);
        exit(1);
    }
    // 4. Close device
    close(fd);
    // 5. Print the message to stdout (=1)
    if (write(1, msg, b_num) <= 0){
        fprintf(stderr, "print message failed. Error number: %s\n", strerror(errno));
        exit(1);
    }
    // 6. Exit
    exit(0);
}
