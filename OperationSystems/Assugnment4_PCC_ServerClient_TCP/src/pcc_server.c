#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>	// for sigaction(), SIG_INT, sa
#include <stdint.h>		// For uint32_t = unsigned 32-bit integer type

#define PRINTABLE_CHARS (95)
#define PADDING_CLIENTS (10)
#define BUFFER_LEN (1048576)    // 1 MB buffer length


int connection_flag = 0;
int sigint_flag     = 0;
uint32_t pcc_total[PRINTABLE_CHARS];  // structure for couters pritable chars

void print_pcc_total();

static void sigint_handler(int sig){
    sigint_flag = 1;  // rais singint_flag to indicate that sigint occured.

    if (connection_flag == 0){	// if the server wait to new connection while sigint
    	print_pcc_total();
    	exit(0);
    }
}

void print_pcc_total(){
	// Output the statistics of all pritable characters 
	int i = 0;
	for(i = 0; i < PRINTABLE_CHARS; i++){
        fprintf(stdout, "char '%c' : %u times\n", (char)(i+32), pcc_total[i]);
    }
}

int set_sigint(){
	// Set new handler to sigint
    struct sigaction sa;
	sa.sa_handler = sigint_handler;
	sa.sa_flags = SA_RESTART;
	if (sigaction(SIGINT, &sa, NULL) == -1) {
        return -1;
    }

    return 0;
}

int main(int argc, char *argv[])
{
    int listenfd  = -1;
    int connfd    = -1;
	uint32_t temp_cnt_char[PRINTABLE_CHARS];
	
    unsigned short  server_port;

    struct sockaddr_in serv_addr;	/// *** From recitation
    struct sockaddr_in peer_addr;	/// *** From recitation
    socklen_t addrsize = sizeof(struct sockaddr_in );	/// *** From recitation

    if (argc != 2){
        // Error in number of arguments
        errno = EINVAL;
        fprintf(stderr, "Error: Arguments Failed. %s\n", strerror(errno));
        return 1;
    }
    
    server_port = (unsigned short)atoi(argv[1]);
	
	// ================== Set SIGINT handler ============================================
    if (set_sigint() == -1){
        // Error in handle sigint
        fprintf(stderr, "Error: SIGINT Handle Failed. %s\n", strerror(errno));
        return 1;
    }

	// ================== Create socket & bind & listen =================================
    if((listenfd = socket(AF_INET, SOCK_STREAM, 0)) == -1){
    	fprintf(stderr, "Error: Create Socket Failed. %s\n", strerror(errno));
        return 1;
    }
    
    if (setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &(int){1}, sizeof(int)) < 0){
    	fprintf(stderr, "Error: setsockopt(SO_REUSEADDR) Failed. %s\n", strerror(errno));
    	return 1;
    }
    
    /// From recitation from here
    memset(&serv_addr, 0, addrsize);
	
    serv_addr.sin_family = AF_INET;
    // INADDR_ANY = any local machine address
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(server_port);

    if(0 != bind(listenfd, (struct sockaddr*)&serv_addr, addrsize)){
        fprintf(stderr, "Error : Bind Socket Failed. %s \n", strerror(errno));
        return 1;
    }
	
    if(0 != listen(listenfd, PADDING_CLIENTS)){
        fprintf(stderr, "Error : Listen Socket Failed. %s \n", strerror(errno));
        return 1;
    }
	/// From recitation until here
	
	// ================== Accept connection loop =========================================
    while(1){
        // After sigint, stop accept new sockets
        if (sigint_flag == 1){
            break;
        }
        
        memset(temp_cnt_char, 0, sizeof(uint32_t)*PRINTABLE_CHARS);	// initialized temp_cnt_char
        
        connfd = accept(listenfd, (struct sockaddr*) &peer_addr, &addrsize);
        if(connfd < 0){
            fprintf(stderr, "Error : Accept Socket Failed. %s \n", strerror(errno));
            return 1;
        }

        connection_flag = 1;    // connect has been made

        // ============== Read number of bytes (N) from client ============================ 
        int bytes_read = 0; 	
       	uint32_t N = 0;			// Number of bytes to read from client
       
        bytes_read = read(connfd, &N, 4);
        if(bytes_read <= 0){
            // Error in read
            if (errno == ETIMEDOUT || errno == ECONNRESET || errno == EPIPE || bytes_read == 0){
	        	connection_flag = 0; 
	        	continue;
        	}
            fprintf(stderr, "Error : Read from client Failed. %s \n", strerror(errno));
            return 1;
        }
        N = ntohl(N);	// bytes network order

        // ============== Read client's file content - N bytes overall ==================== 
        bytes_read = 0;
        char recv_buff[BUFFER_LEN];
        int count_recv_btyes = 0;
 		int error_flag = 0;
        int i = 0;
        uint32_t count_printable = 0; // counter for printable charcter of client
        int num_bytes_to_read;
        while(count_recv_btyes < N)
        {
        	if (N - count_recv_btyes < BUFFER_LEN){
        		num_bytes_to_read = N - count_recv_btyes;
        	}
        	else{
        		num_bytes_to_read = BUFFER_LEN;
        	}
        	
            bytes_read = read(connfd, recv_buff, num_bytes_to_read);
            if (bytes_read <= 0){
            	if (errno == ETIMEDOUT || errno == ECONNRESET || errno == EPIPE || bytes_read == 0){
	        		error_flag = 1;
	        		break;
        		}
            	fprintf(stderr, "Error : Read from client Failed. %s \n", strerror(errno));
            	return 1;
            }
            
            count_recv_btyes += bytes_read;
             
            for (i = 0; i < bytes_read; i++){
                if (recv_buff[i] >= 32 && recv_buff[i] <= 126){
                    count_printable++;
                    temp_cnt_char[recv_buff[i] - 32]++;
                }
            }
        }
        if (error_flag == 1){
        	connection_flag = 0;
        	close(connfd); 
        	continue;
        }

        // ============== Write client count of printable characters ======================
        
        uint32_t count_network = htonl(count_printable);	// bytes network order
        int nsent = 0;
        nsent = write(connfd, &count_network, sizeof(uint32_t));
        if(nsent <= 0){
            // Error in sending countable characters
            if (errno == ETIMEDOUT || errno == ECONNRESET || errno == EPIPE || nsent == 0){
            	close(connfd);
            	connection_flag = 0;
	        	continue;
        	}
            fprintf(stderr, "Error : Write to client Failed. %s \n", strerror(errno));
            return 1;
        }

        // close socket
        close(connfd);
        
        // ============== Update pcc_total ===============================================
        
        for (i = 0; i < PRINTABLE_CHARS; i++){
            pcc_total[i]+= temp_cnt_char[i];
        }

        connection_flag = 0;
    }
    
  	print_pcc_total();
    exit(1);
}
