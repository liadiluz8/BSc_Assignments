#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h> /* For dup2() sys call */
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>

// arglist: a list of char* arguments (words) provided by the user
// it contains count+1 items, where the last item (arglist[count]) and *only* the last is NULL
// RETURNS -1 if should continue, 0 otherwise
int process_arglist(int count, char **arglist);

// prepare and finalize calls for initialization and destruction of anything required
int prepare(void);
int finalize(void);


// Eran's Trick for treating Zonbies
// From: https://stackoverflow.com/questions/7171722/how-can-i-handle-sigchld/7171836#7171836
static void child_handler(int sig) {
    pid_t wait_res;
    int status;
	
    /* EEEEXTEERMINAAATE! */
    while((wait_res = waitpid(-1, &status, WNOHANG)) > 0);
	
    if(wait_res == -1 && errno != ECHILD && errno != EINTR){
        // Handling error while wait
        perror("Error in wait for treating zombies");
        exit(1);
    }
}

int set_sigint_ignore() {
	struct sigaction sa;
	sa.sa_handler = SIG_IGN;
	sa.sa_flags = SA_RESTART;
	if (sigaction(SIGINT, &sa, NULL) == -1) {
        return -1;
    }

    return 0;
}

int set_sigint_default() {
	struct sigaction sa;
	sa.sa_handler = SIG_DFL;
	sa.sa_flags = SA_RESTART;
	if (sigaction(SIGINT, &sa, NULL) == -1) {
        return -1;
    }

    return 0;
}

int prepare() {
    // Erans's Trick continue: Treating zombies
    /* Establish handler. */
    
    struct sigaction sa;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sa.sa_handler = child_handler;

    if (sigaction(SIGCHLD, &sa, NULL) == -1){
        // Error in sigaction
        perror("Error while sigaction on SIGCHLD");
        return 1;
    }
    
    // Set parent process to ignore sigint
    if (set_sigint_ignore() == -1){
        // Error in sigaction
        perror("Error while sigaction on SIG_IGN");
        return 1;
    }

    return 0;
}

int extract_operation(int count, char **arglist, int *op_num, int *cnt_arg_1){
    int i = 0;

    for (i = 0; i < count; i++) {
        if (strcmp(arglist[i], "&") == 0) {
            (*op_num) = 1;
            return 1;
        } else if (strcmp(arglist[i], "|") == 0) {
            (*op_num) = 2;
            return 1;
        } else if (strcmp(arglist[i], ">") == 0) {
            (*op_num) = 3;
            return 1;
        } else {
            (*cnt_arg_1)++;
        }
    }

    return 1;
}

int process_arglist(int count, char **arglist) {
    int op_number = 0;
    int count_arg_1 = 0;
    char **args1 = NULL;
    char **args2 = NULL;
    char *command = NULL, *command2 = NULL;
    pid_t session_id;

    int ext_op_ret = extract_operation(count, arglist, &op_number, &count_arg_1);
    if (ext_op_ret == 0){
        // Error while extracting
    }

    if (op_number == 0 || op_number == 1){
        /* 1. Executing commands & 2. Executing commands in the background */
        command = arglist[0];
        session_id = fork();
        if (session_id == -1) {
            /* Error handling */
            perror("Error while fork");
            return 0;
        } else if (session_id == 0) {
            /* Child */
            // Signal handler: Set the background procces to ignore SIGINT and the foreground process to get SIGINT
            
            
            if (op_number == 0){
                if (set_sigint_default() == -1){
                    // Error in sigaction
                    perror("Error while sigaction on SIG_DFL");
                    exit(1);
                }
            }
            else{
            	if (set_sigint_ignore() == -1){
                    // Error in sigaction
                    perror("Error while sigaction on SIG_IGN");
                    exit(1);
                }
            }
            

            if (op_number == 1){
                arglist[count_arg_1] = NULL; // Assign NULL instead of &
            }
            args1 = arglist;

            int exe_ret = execvp(command, args1);
            if (exe_ret == -1) {
                /* failure */
                perror("Error while execvp in child");
                exit(1);    // exit 1 or 0??
            }
        } else {
            /* Parent */
            if (op_number == 0){
                // 1. Executing commands : Wait until child process will finish
                int status = 0;
                pid_t wait_res = waitpid(session_id, &status, 0);
                if (wait_res == -1 && errno != ECHILD && errno != EINTR){
                    // Error
                    perror("Error while wait in parent");
                    return 0;
                }
            }
        }
    }
    else if (op_number == 2){
        /* 3. Single piping */
        pid_t session_id1, session_id2;
        // Extract two lists arguments from arglist array
        arglist[count_arg_1] = NULL;
        args1 = arglist;
        args2 = arglist + count_arg_1 + 1;

        int pipefd[2];
        int pipe_ret = pipe(pipefd);
        if (pipe_ret == -1) {
            // Error handling
            perror("pipe");
            exit(-1);
        }

        command = args1[0];
        command2 = args2[0];

        session_id1 = fork();
        if (session_id1 == -1) {
            /* Error handling */
            perror("Error while fork");
            close(pipefd[0]);
            close(pipefd[1]);
            return 0;
        } else if (session_id1 == 0) {
            /* Child 1 : Writer*/
            // Signal handler: Set the foreground process SIGINT as default
            if (set_sigint_default() == -1){
                // Error in sigaction
                perror("Error while sigaction on SIG_DFL");
                exit(1);
            }
            // Redirect the output of the Writer to the fd_pipe of the Reader
            // Writer process StdOut = fd_pipe write, 1 = StdOut of process
            if (dup2(pipefd[1], 1) == -1){
                perror("Error while dup2");
                exit(1);
            } 

            // Close unused read end
            close(pipefd[0]);

            int exe_ret = execvp(command, args1);
            if (exe_ret == -1) {
                /* failure */
                perror("Error while execvp in child");
                close(pipefd[1]);
                exit(1);
            }
        } else {
            /* Parent */
            session_id2 = fork();
            if (session_id2 == -1) {
                /* Error handling */
                perror("Error while fork");
                close(pipefd[0]);
                close(pipefd[1]);
                return 0;
            } else if (session_id2 == 0) {
                /* Child 2 : Reader*/
                // Signal handler: Set the foreground process SIGINT as default
                if (set_sigint_default() == -1){
                    // Error in sigaction
                    perror("Error while sigaction on SIG_DFL");
                    exit(1);
                }

                // Redirect the input of the reader to the fd_pipe of the Reader
                // Reader process StdIn = fd_pipe of Reader, 0 = StdIn of process
                if(dup2(pipefd[0], 0) == -1){
                    perror("Error while dup2");
                    exit(1);
                }

                // Close unused write end
                close(pipefd[1]);

                int exe_ret = execvp(command2, args2);
                if (exe_ret == -1) {
                    /* failure */
                    perror("Error while execvp in child");
                    close(pipefd[0]);
                    exit(1);
                }
            } else {
                /* Parent */
                // close pipe for parent process (are not relevant to the pipe)
                close(pipefd[0]);
                close(pipefd[1]);

                int status = 0, status2 = 0;
                // Parent waits until both children process are finish
                // option > 1 implies wait to any process with the specific id
                pid_t pid1 = waitpid(session_id1, &status, 0);
                if (pid1 == -1 && errno != ECHILD && errno != EINTR){
                    // Error
                    perror("Error while wait in parent");
                    return 0;
                }
                pid_t pid2 = waitpid(session_id2, &status2, 0);
                if (pid2 == -1 && errno != ECHILD && errno != EINTR){
                    // Error
                    perror("Error while wait in parent");
                    return 0;
                }
            }
        }
    }
    else if (op_number == 3){
        /* 4. Output redirecting */
        const char *output_file_name = arglist[count - 1]; // The last argument of arglist is output file name
        arglist[count_arg_1] = NULL; // Set NULL instead of '>'
        args1 = arglist;
        command = args1[0];

        // Open file as file directory for:
        // Write-only, Create if the file does not exist and Trunc content if exist.
        // The permissions S_IRWXU are Read, Write and Executable.
        int fd_file = open(output_file_name, O_CREAT | O_TRUNC | O_WRONLY, S_IRWXU);
        if (fd_file < 0) {
            /* Open file failed */
            perror("Error while try open a file");
            return 0;
        }

        session_id = fork();
        if (session_id == -1) {
            /* Error handling */
            perror("Error while fork");
            return 0;
        } else if (session_id == 0) {
            /* Child */
            // Signal handler: Set the foreground process SIGINT as default
            if (set_sigint_default() == -1){
                // Error in sigaction
                perror("Error while sigaction on SIG_DFL");
                exit(1);
            }

            // Redirect the StdOut to fd of the file
            if(dup2(fd_file, 1) == -1){
                perror("Error while dup2");
                exit(1);
            }

            int exe_ret = execvp(command, args1);
            if (exe_ret == -1) {
                /* failure */
                perror("Error while execvp in child");
                exit(1);
            }
        } else {
            /* Parent */
            // Parent wait until child end
            int status = 0;
            pid_t pid = waitpid(session_id, &status, 0);
            if (pid == -1 && errno != ECHILD && errno != EINTR){
                // Error in wait
                perror("Error while wait in parent");
                return 0;
            }

            close(fd_file);
        }
    }
    
    return 1;
}


int finalize()
{
    return 0;
}
