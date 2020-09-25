#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

int main(int argc, char **argv)
{
    // Exit program if there are no required arguments
    if (argc != 4)
    {
        printf("Usage: %s <ncores/node> <dataSize> <mode> \n", argv[0]);
        return -1;
    }
    
    int ncores_per_node = atoi(argv[1]);
    int dataSize = atoi(argv[2]);
    int mode = atoi(argv[3]);
    
    int rank = 0;
    int process_count = 0;
    
    // MPI Initial
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
        printf("ERROR: MPI_Init error\n");
    if (MPI_Comm_size(MPI_COMM_WORLD, &process_count) != MPI_SUCCESS)
        printf("ERROR: MPI_Comm_size error\n");
    if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS)
        printf("ERROR: MPI_Comm_rank error\n");
    
    // initial buffer with rank id (e.g., rank 0: 00000..)
    unsigned char* buffer = malloc(dataSize);
    int data_num = dataSize/sizeof(int);
    for (int i = 0; i < data_num; i++)
        memcpy(&buffer[i*sizeof(int)], &rank, sizeof(int));
    
    // decide aggregators
    int is_aggregator = (rank % ncores_per_node == 0)? 1: 0;
    int node_id = rank / ncores_per_node;
    
    // mallco receive buffer for aggregators
    unsigned char* recv_buffer = NULL;
    if (is_aggregator == 1)
        recv_buffer = malloc(dataSize * ncores_per_node);
    
    int node_num = process_count / ncores_per_node;  // the number of nodes
    if (process_count % ncores_per_node > 0)
        node_num += 1;
    
    // run 20 iteration
    double avg_it_time = 0;
    double max_it_time = 0;
    for (int it = 0; it < 20; it++)
    {
        double communication_start = MPI_Wtime();
        // MPI request and status
        MPI_Request req[ncores_per_node + 1];
        MPI_Status stat[ncores_per_node + 1];
        int req_num = 0;
        if (mode == 0)  // This case aggregation will not cross node boundary
        {
            // Received data from others
            if (is_aggregator == 1) // rank 0 64 128 192
            {
                for (int i = 0; i < ncores_per_node; i++)
                {
                    int recv_rank = i + rank; // 0  i(0-63)
                    MPI_Irecv(&recv_buffer[i * dataSize], dataSize, MPI_UNSIGNED_CHAR, recv_rank, recv_rank, MPI_COMM_WORLD, &req[req_num]);
                    req_num++;
                }
            }
            
            // Send data to aggregators
            int send_rank = node_id * ncores_per_node;
            MPI_Isend(buffer, dataSize, MPI_UNSIGNED_CHAR, send_rank, rank,
                      MPI_COMM_WORLD, &req[req_num]);
            req_num++;
        }
        else if (mode == 1) // This case aggregation will cross node boundary
        {
            // Received data from others
            if (is_aggregator == 1)
            {
                // the data is received by the previous node, e.g., (1:64-127)->0
                int recv_node = ((node_id - 1) > -1)? (node_id - 1): (node_num - 1);
                for (int i = 0; i < ncores_per_node; i++)
                {
                    // the aggregator received data from which ranks, e.g., (64)<-(i:0-63 + 0*64)
                    int recv_rank = i + recv_node * ncores_per_node;
                    MPI_Irecv(&recv_buffer[i * dataSize], dataSize, MPI_UNSIGNED_CHAR, recv_rank, recv_rank, MPI_COMM_WORLD, &req[req_num]);
                    req_num++;
                }
            }
            
            // Send data to aggregators
            int send_node = ((node_id + 1) < node_num)? (node_id + 1): 0;// which node to send, e.g., (0:0-63)->1
            int send_rank =  send_node * ncores_per_node; // send to which aggregator, e.g., (0:0-63)->64
            MPI_Isend(buffer, dataSize, MPI_UNSIGNED_CHAR, send_rank, rank,
                      MPI_COMM_WORLD, &req[req_num]);
            req_num++;
        }
        else if (mode == 2) // Round Robin
        {
            // group size: divided the processes on a node into N group (N: the number of nodes), e.g., 64/4 = 16
            int node_div = ncores_per_node / node_num;
            
            // Received data from others
            if (is_aggregator == 1)
            {
                int relative_id = node_id * node_div;  // (0:0, 1:16, 2:32, 3:64)
                for (int i = 0; i < node_num; i++)
                {
                    for (int j = 0; j < node_div; j++)
                    {
                        // node 1 (64): (i:0-3)*64+16+(0:15)->((16:31),(80:95),(144:159),(208:223))
                        int recv_rank = i * ncores_per_node + relative_id + j;
                        int index = (i * node_div + j);
                        MPI_Irecv(&recv_buffer[index * dataSize], dataSize, MPI_UNSIGNED_CHAR, recv_rank, recv_rank, MPI_COMM_WORLD, &req[req_num]);
                        req_num++;
                    }
                }
            }
            
            // Send data to aggregators
            int send_rank = (rank - node_id * ncores_per_node) / node_div * ncores_per_node;
            MPI_Isend(buffer, dataSize, MPI_UNSIGNED_CHAR, send_rank, rank,
                      MPI_COMM_WORLD, &req[req_num]);
            req_num++;
        }
        else
        {
            printf("ERROR: unsupported mode!");
        }
        MPI_Waitall(req_num, req, stat);
        
        double communication_end = MPI_Wtime();
        double communication_time = communication_end - communication_start;
        
        avg_it_time += communication_time/20;
        max_it_time = (communication_time > max_it_time)? communication_time: max_it_time;
        
//        if (rank == 0)
//        {
//            printf("The current iteration is %d.\nThe current mode is %d.\nThe transfered data is %d.\nThe communication time is %f.\n", it, mode, dataSize, communication_time);
//        }
    }
    
    double max_time = 0;
    double avg_time = 0;
    MPI_Reduce(&max_it_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&avg_it_time, &avg_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("The current mode is %d.\nThe transfered data is %d.\nThe maximum communication time is %f.\nThe average communication time is %f.\n", mode, dataSize, max_time, avg_time);
    }
//    if (is_aggregator == 1)
//    {
//        int size = dataSize * ncores_per_node / sizeof(int);
//        for (int i = 0; i < size; i++)
//        {
//            int a;
//            memcpy(&a, &recv_buffer[i*sizeof(int)], sizeof(int));
//            printf("%d, %d\n", rank, a);
//        }
//    }
    
    free(buffer);
    free(recv_buffer);
    MPI_Finalize();
    return 0;
}

