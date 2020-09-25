#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    
    unsigned char* buffer = malloc(dataSize);
    
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
    
    int node_num = ceil((float)process_count/ncores_per_node);
    
    double communication_start = MPI_Wtime();
    // MPI request and status
    MPI_Request req[ncores_per_node + 1];
    MPI_Status stat[ncores_per_node + 1];
    int req_num = 0;
    if (mode == 0)  // This case aggregation will not cross node boundary
    {
        // Send data to aggregators
        int send_rank = node_id * ncores_per_node;
        MPI_Isend(buffer, dataSize, MPI_UNSIGNED_CHAR, send_rank, rank,
                  MPI_COMM_WORLD, &req[req_num]);
        req_num++;
        
        // Received data from others
        if (is_aggregator == 1)
        {
            for (int i = 0; i < ncores_per_node; i++)
            {
                int recv_rank = i + rank;
                MPI_Irecv(&recv_buffer[i * dataSize], dataSize, MPI_UNSIGNED_CHAR, recv_rank, recv_rank, MPI_COMM_WORLD, &req[req_num]);
                req_num++;
            }
        }
    }
    else if (mode == 1) // This case aggregation will cross node boundary
    {
        int send_node = ((node_id + 1) < node_num)? (node_id + 1): 0;
        int send_rank =  send_node * ncores_per_node;
        
        MPI_Isend(buffer, dataSize, MPI_UNSIGNED_CHAR, send_rank, rank,
                  MPI_COMM_WORLD, &req[req_num]);
        req_num++;
        
        // Received data from others
        if (is_aggregator == 1)
        {
            int agg_id = rank / ncores_per_node;
            int recv_node = ((agg_id - 1) > -1)? (agg_id - 1): (node_num - 1);
            for (int i = 0; i < ncores_per_node; i++)
            {
                int recv_rank = i + recv_node * ncores_per_node;
                MPI_Irecv(&recv_buffer[i * dataSize], dataSize, MPI_UNSIGNED_CHAR, recv_rank, recv_rank, MPI_COMM_WORLD, &req[req_num]);
                req_num++;
            }
        }
    }
    else if (mode == 2) // Round Robin
    {
        int node_div = ncores_per_node / node_num;
        int send_rank = (rank - node_id * ncores_per_node) / node_div * ncores_per_node;
        MPI_Isend(buffer, dataSize, MPI_UNSIGNED_CHAR, send_rank, rank,
                  MPI_COMM_WORLD, &req[req_num]);
        req_num++;
        
        if (is_aggregator == 1)
        {
            int relative_id = node_id * node_div;
            for (int i = 0; i < node_num; i++)
            {
                for (int j = 0; j < node_div; j++)
                {
                    int recv_rank = i * ncores_per_node + relative_id + j;
                    int index = (i * node_div + j);
                    MPI_Irecv(&recv_buffer[index * dataSize], dataSize, MPI_UNSIGNED_CHAR, recv_rank, recv_rank, MPI_COMM_WORLD, &req[req_num]);
                    req_num++;
                }
            }
        }
    }
    else
    {
        printf("ERROR: unsupported mode!");
    }
    MPI_Waitall(req_num, req, stat);
    double communication_end = MPI_Wtime();
    
    double communication_time = communication_end - communication_start;
    double max_time = 0;
    MPI_Reduce(&communication_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0)
    {
        printf("The current mode is %d.\nThe transfered data is %d.\nThe communication time is %f.\n", mode, dataSize, max_time);
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
}

