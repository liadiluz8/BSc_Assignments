// *** here preproccing declrations from recitation
// Declare what kind of code we want
// from the header files. Defining __KERNEL__
// and MODULE allows us to access kernel-level
// code not usually available to userspace programs.
#undef __KERNEL__
#define __KERNEL__
#undef MODULE
#define MODULE

#include <linux/slab.h>     /* kfree() and kmalloc()*/
#include <linux/kernel.h>   /* We're doing kernel work */
#include <linux/module.h>   /* Specifically, a module */
#include <linux/fs.h>       /* for register_chrdev */
#include <linux/uaccess.h>  /* for get_user and put_user */
#include <linux/string.h>   /* for memset. NOTE - not string.h!*/

MODULE_LICENSE("GPL");

//Our custom definitions of IOCTL operations
#include "message_slot.h"
#define MAX_DEVICES (257)

// struct for driver devices
// Array of devices, the i'th cell represent the i'th minor number llist - 
//                  all the channels with this minor number
device_channels_list device_files[MAX_DEVICES];  

// *** Function names based on recitation examples
//================== DEVICE FUNCTIONS ===========================
static int device_open(struct inode* inode, struct file* file) {
    int minor_number;
    channel_file* chl_file;
	      
    if (inode == NULL || file == NULL){
        return -EINVAL;
    }
    
    printk( "Invocing device_open(%p,%p)\n" ,inode, file );
    
    minor_number = iminor(inode);
	if (minor_number < 0 || minor_number >= MAX_DEVICES){
		// Minor number excceed 256
		return -EIO;
	} 
	
    chl_file = NULL;
    chl_file = (channel_file*)kmalloc(sizeof(channel_file), GFP_KERNEL);
    if(chl_file == NULL){
        // Error while kmalloc
        return -ENOMEM;
    }

    chl_file->minor_num = minor_number;
    chl_file->channel_id = 0;
    chl_file->curr_channel = NULL;
	
    file->private_data = (void*)(chl_file);
	
	device_files[minor_number].minor_num = minor_number;
	
    return SUCCESS;
}

//---------------------------------------------------------------
static int device_release(struct inode* inode, struct file* file) {
    if (inode == NULL || file == NULL){
        return -EINVAL;
    }

    printk("Invoking device_release(%p,%p)\n", inode, file);
    
    if (file->private_data != NULL){
        kfree(file->private_data);
    }
    return SUCCESS;
}

//---------------------------------------------------------------
// a process which has already opened
// the device file attempts to read from it
static ssize_t device_read(struct file* file, char __user* buffer, size_t length, loff_t* offset){
    int msg_size, b_num = 0;
    channel_file* chl_file;
    channel* curr_channel;
    char *msg;

    if (file == NULL || buffer == NULL){
        return -EINVAL;
    }
	
	printk( "Invocing device_read(%p,%p,%zu)\n", file, buffer, length );
          
    chl_file = (channel_file*)(file->private_data);
    if (chl_file == NULL){
        // Error in channel file: no channel set
        return -EINVAL;
    }

    // Extract current chanell from file
    curr_channel = chl_file->curr_channel;
    if (curr_channel == NULL || curr_channel->channel_id == 0){
        // Error in channel
        return -EINVAL;
    }

    msg_size = curr_channel->msg_size;
    b_num = 0;
    msg = curr_channel->msg;
    if (msg_size == 0 || msg == NULL){
        // Message does not exist on channel
        return -EWOULDBLOCK;
    }
    
    if (msg_size > length){
        // Error in message size
        return -ENOSPC; 	//  No space left on device
    }

    // put msg to buffer
    for (b_num = 0; b_num < msg_size; b_num++){
        if(put_user(msg[b_num], buffer+b_num) != 0){
            // Error during put_user
            return -EIO;	// error input output
        }
    }
    printk( "device_read: Successfuly read the message %s in size %d\n", buffer, msg_size );
    // return the number of bits that are read
    return msg_size;
}

//---------------------------------------------------------------
// a process which has already opened
// the device file attempts to write to it
static ssize_t device_write(struct file* file, const char __user* buffer, size_t length, loff_t* offset){
    int i = 0, b_num = 0;
    char msg[BUF_LEN];
    channel_file* chl_file;
    channel* curr_channel;

    if (file == NULL || buffer == NULL){
        return -EINVAL;
    }
	
	printk("Invoking device_write(%p,%s,%zu)\n", file, buffer, length);
	
    chl_file = (channel_file*)(file->private_data);
    if (chl_file == NULL){
        // Error in channel file
        return -EINVAL;
    }

    // Extract current chanell from file
    curr_channel = chl_file->curr_channel;
    
    
    if (length == 0 || BUF_LEN < length){
        // Error in length of message
        return -EMSGSIZE;
    }

    if (curr_channel == NULL || curr_channel->channel_id == 0){
        // Error in channel
        return -EINVAL;
    }

    // Get the message from user - Atomic read
    b_num = 0;
    for (b_num = 0; b_num < length; b_num++){
        if (get_user(msg[b_num], &buffer[b_num]) != 0){
            // Error while reading the message
            return -EIO;
        }
    }

    // put the message on current channel
    curr_channel->msg_size = length;
    i = 0;
    for (i = 0; i < length; i++){
        (curr_channel->msg)[i] = msg[i];
    }
    
    printk( "device_write: Successfuly write the message %s in size %zu\n", buffer, length );
    
    // return the number of bits that are written
    return length;
}

//----------------------------------------------------------------
static long device_ioctl(struct file* file, unsigned int ioctl_command_id, unsigned long ioctl_param){
    unsigned int minor_num;
    channel *head, *temp;
    channel_file* chl_file;
	
	printk( "Invocing device_ioctl(%p,%u,%lu)\n", file, ioctl_command_id, ioctl_param );
	
    if(file == NULL || ioctl_command_id != MSG_SLOT_CHANNEL || ioctl_param == 0){
        // Error in arguments
        return -EINVAL;
    }
    
    chl_file = (channel_file*)(file->private_data);
    if (chl_file == NULL){
        // Error in channel file
        return -ENODEV;    // no such device
    }
	
    // Extract minor number from file
    minor_num = chl_file->minor_num;
    
    if (minor_num == -1){
        // unopened file
        return -EIO;
    }

    chl_file->channel_id = (unsigned int)ioctl_param;

    head = NULL;
    temp = NULL;
    head = device_files[minor_num].head;

    if (head == NULL){  // If there is no channel in given minor number (device file)
        head = (channel*)kmalloc(sizeof(channel), GFP_KERNEL);
        if (head == NULL){
            // Error while kmallok
            return -ENOMEM;
        }
        head->channel_id = ioctl_param;
        head->msg_size = 0;
        head->next = NULL;

        // Set the new head in the array od devices
        device_files[minor_num].head = head;
    }

    // Find the given channel by ioctl_param
    while(head->next != NULL){
        if(head->channel_id == ioctl_param){
            break;
        }
        head = head->next;
    }
    if (head->channel_id != ioctl_param){   // there is no such channel_id as 
                                            //      ioctl_param in device_file
        // Create new channel with ioctl_param as id
        temp = (channel*)kmalloc(sizeof(channel), GFP_KERNEL);
        if (temp == NULL){
            // Error while kmallok
            return -ENOMEM;
        }
        temp->channel_id = ioctl_param;
        temp->msg_size = 0;
        temp->next = NULL;

        // head is the last channel in device file (minor number)
        // We add the new channel temp at the end
        head->next = temp;      
        head = head->next;
    }

    // head is the channel of ioctl_param as channel_id
    chl_file->curr_channel = head;
	
    return SUCCESS;
}

// *** Devise setup Based on recitation
//==================== DEVICE SETUP =============================

// This structure will hold the functions to be called
// when a process does something to the device we created
struct file_operations Fops = {
        .owner	        = THIS_MODULE,
        .read           = device_read,
        .write          = device_write,
        .open           = device_open,
        .unlocked_ioctl = device_ioctl,
        .release        = device_release,
};

//---------------------------------------------------------------
// Initialize the module - Register the character device
static int __init msg_slot_init(void) {
    int rc = -1, i = 0;
    // Register driver capabilities. Obtain major num
    rc = register_chrdev(MAJOR_NUM, DEVICE_RANGE_NAME, &Fops);

    // Negative values signify an error
    if( rc < 0 ) {
        printk(KERN_ALERT "%s registration failed for  %d\n", DEVICE_RANGE_NAME, MAJOR_NUM);
        return rc;
    }

    // Init the data structure for channels
    i = 0;
    for (i = 0; i < MAX_DEVICES; i++){
        device_files[i].minor_num = -1;
        device_files[i].head = NULL;
    }
	
	printk( "Registeration is successful. "
          "The major device number is %d.\n", MAJOR_NUM );
	printk( "If you want to talk to the device driver,\n" );
	printk( "you have to create a device file:\n" );
	printk( "mknod /dev/your_device_file_name c %d 0\n", MAJOR_NUM );
	printk( "You can read and sent messages through the copiled files of message_reader.c and message_sender.c.\n" );
    printk( "message_sender.c takes 3 arguments: your_device_file_name, channel id and your message.\n" );
    printk( "the sender writes your message to the specified channel in the device file you supplied.\n" );
	printk( "message_sender.c takes 2 arguments: your_device_file_name and channel id.\n" );
    printk( "the reader reads the last message from the specified channel in the device file you supplied and prints it to stdout.\n" );
	printk( "Dont forget to rm the device file and "
          "rmmod when you're done\n" );
          
    return SUCCESS;
}

//---------------------------------------------------------------
static void __exit msg_slot_cleanup(void) {
    // Unregister the device
    // Should always succeed

    // Free all allocated memory (channels data structure)
    int i = 0;
    channel* chl = NULL;
    channel* temp = NULL;

    for(i = 0; i < MAX_DEVICES; i++){
        chl = device_files[i].head;

        while (chl != NULL){
            temp = chl->next;
            kfree(chl);  // free channel struct
            chl = temp;
        }
    }
	printk( "Exit modul and clean up");
    unregister_chrdev(MAJOR_NUM, DEVICE_RANGE_NAME);
}


//---------------------------------------------------------------
module_init(msg_slot_init);
module_exit(msg_slot_cleanup);

//========================= END OF FILE =========================
