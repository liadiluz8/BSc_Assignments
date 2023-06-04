#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <errno.h>
#include <stdint.h>	// For uint32_t = unsigned 32-bit integer type

#define BUFFER_LEN (1048576)    // 1 MB buffer length

int main(int argc, char *argv[])
{
    int sockfd = -1;
    struct sockaddr_in serv_addr; //// *** From recitation
    char *server_ip, *server_port, *file_path; 

    if(argc != 4){
        // Error in number of arguments
        errno = EINVAL;
        fprintf(stderr, "Error: Arguments. %s\n", strerror(errno));
        return 1;
    }

    server_ip     = argv[1];
    server_port   = argv[2];
    file_path     = argv[3];

	// ================== Create socket & Connect to server =============================
    if((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0){
        fprintf(stderr, "Error: Create Socket Failed. %s\n", strerror(errno));
        return 1;
    }

    // Connection to server
    memset(&serv_addr, 0, sizeof(serv_addr));	/// *** From recitation
    serv_addr.sin_family = AF_INET;				/// *** From recitation
    serv_addr.sin_port = htons((unsigned short)atoi(server_port)); 	// htons for endiannes
    
    // Convert IP from string to bytes
    if(inet_pton(AF_INET, server_ip, &(serv_addr.sin_addr.s_addr)) != 1){
    	fprintf(stderr, "Error : Connect Failed. %s \n", strerror(errno));
        return 1;
    }

    // Connect socket to target address
    if(connect(sockfd, (struct sockaddr*) &serv_addr, sizeof(serv_addr)) < 0){
        fprintf(stderr, "Error : Connect Failed. %s \n", strerror(errno));
        return 1;
    }

    // ================== Open file =======================================================
    FILE *fp = NULL;
    fp = fopen(file_path, "r");
    if (fp == NULL){
        // Error in opening file
        errno = ENOENT;
        fprintf(stderr, "Error : Open File Failed. %s \n", strerror(errno));
        return 1;
    }
    fseek(fp, 0L, SEEK_END);
    uint32_t N             = ftell(fp);	  	// number of bytes of the file
    uint32_t N_network_f   = htonl(N);		// N in network order format     
    rewind(fp);
	
	// ================ Send to server N=length of file ===================================
    int nsent = 0;
    nsent = write(sockfd, &N_network_f, 4);
    if(nsent <= 0){
        // Error in sending N
        fprintf(stderr, "Error : Write to server Failed. %s \n", strerror(errno));
        return 1;
    }

  	// ================ Send to server the file content (N bytes) =========================
    nsent = 0;
    char    file_buffer[BUFFER_LEN];
    size_t  read_elements   = 0;
    int     totalsent       = 0;
    int     notwritten      = 0;
	
    while ((read_elements = fread(file_buffer, 1, BUFFER_LEN, fp)) > 0){
        // SEND file_buffer to server
        notwritten = read_elements;
		totalsent = 0;
        
        while(notwritten > 0){
            nsent = write(sockfd, file_buffer + totalsent, notwritten);
            if (nsent <= 0){
                // Error in sending to server
                fprintf(stderr, "Error : Write to server Failed. %s \n", strerror(errno));
		        return 1;
            }
            	
            totalsent  += nsent;
            notwritten -= nsent;
        }
    }
	fclose(fp);

	// ================ Read number of printable character from server ===================
    int bytes_read = 0;
    uint32_t C = 0;
    
    bytes_read = read(sockfd, &C, sizeof(uint32_t));
    if(bytes_read <= 0){
        // Error in read
        fprintf(stderr, "Error : Read from server Failed. %s \n", strerror(errno));
        return 1;
    }
    
    C = ntohl(C);	// bytes network order
    fprintf(stdout, "# of printable characters: %u\n", C);
    
    close(sockfd);
    return 0;
}
