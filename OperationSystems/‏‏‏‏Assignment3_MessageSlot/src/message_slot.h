// *** Based on recitation example
#ifndef MESSAGE_SLOT_H
#define MESSAGE_SLOT_H

#include <linux/ioctl.h>

#define MAJOR_NUM (235)
#define MSG_SLOT_CHANNEL _IOW(MAJOR_NUM, 0, unsigned int)

#define DEVICE_RANGE_NAME "message_slot"
#define BUF_LEN 128
#define SUCCESS 0

// Struct for channel node
// channel_id is the channek id of the channel
// msg is the message (string in amx length of BUF_LEN)
// msg_size is the effective length of the message
// next is a pointer to the next node (channel)
typedef struct message_slot_channel{
    unsigned int channel_id;
    char msg[BUF_LEN];
    int msg_size;
    struct message_slot_channel* next;
} channel;

// Struct for list of channels. 
// head is a pointer to the head of the list
// minor_num is the minor number that the list refer to.
typedef struct message_slot_channels_list{
    unsigned int minor_num;
    channel* head;
} device_channels_list;

// Struct for file. Contained in privet data of file object.
// minor_num and channel_id are the channel properties in which the file assosiated with.
// curr_channel is a pointer to the channel in which the file assosiated with.
typedef struct channel_file{
    unsigned int minor_num;
    unsigned int channel_id;
    channel* curr_channel;
} channel_file;

#endif