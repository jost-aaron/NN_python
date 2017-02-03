#define input_matrix_access(r, c) (input_matrix[(r)*width + (c)])
#define sum_bridge(r,c) (sum_bridge[(r)*num_sums_per_collum + (c)])

// Constants variable 
  // index 0: number_collums
  // index 1: num_sums_per_collum

__kernel void feed_forward_play(
  	__global int *network_width,
    __global int *sums_per_collum,
    __global float *input_vector,
    __global float *input_matrix, 
    __global float *output_vector,
    __global float *sum_bridge,
    __local float *local_sums)
{
  
    // Get the global position of this instance
    int global_x = get_global_id(0);
    int global_y = get_global_id(1);

    // Load the constants into private memory
    int width = *network_width;
    int num_sums_per_collum = *sums_per_collum;
   
    // Multiply the iput values onto their row
    // For some reason this was transposing the matrix when accesing input_matrix_access so it was switched
    input_matrix_access(global_y,global_x) = input_matrix_access(global_y,global_x) * input_vector[global_y];

    // Find the local id of this instance in its workgroup
    uint local_id = get_local_id(0);

    // Get the work group size
    uint group_size = get_local_size(0);

    // Copy the numbers we want to use from global memory to local memory
    local_sums[local_id] = input_matrix_access(global_y,global_x);

  // Loop for computing local_sums
    for (float stride = group_size/float(2); int(stride)>0; stride /=2)
         {
         // Waiting for each addition into given workgroup
         barrier(CLK_LOCAL_MEM_FENCE);

         // If stride is an odd number add the last value of the current sum to its lift value and store it
         if (!(stride - float(int(stride)) == 0) && local_id == 0){
            local_sums[0] += local_sums[2*uint(stride)];
            stride = float(int(stride));
         }


         barrier(CLK_LOCAL_MEM_FENCE);


          // Divide WorkGroup into 2 parts and add elements 2 by 2
          // between local_id and local_id + stride
          if (local_id < uint(stride))
            local_sums[local_id] += local_sums[local_id + uint(stride)];
         }


         bool single_work_group = true;

         if (single_work_group) {

               // Do this if we dont have multiple workgroups per row
            // Write result into partialSums[nWorkGroups]
            if (local_id == 0) {
              output_vector[global_y] = local_sums[0];
            }

        } else{

      // Wait for all workgroups to get here
      barrier(CLK_GLOBAL_MEM_FENCE);
      // If the workgroup does not contain the row for the sum save it into global memory so we can do another summation
      
        }
 }                  

