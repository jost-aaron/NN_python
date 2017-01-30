#define vec_out_access(r, c) (vec_out[(r)*width + (c)])
#define weights_vec_access(r, c) (weights_vec[(r)*width + (c)])

__kernel void feed_forward(
  	__global int *network_width,
    __global int *num_work_groups_per_row,
    __global int *debug_array,
    __global float *input_vec,
    __global float *weights_vec, 
    __global float *vec_out,
    __local float *localSums)
{
  
    // Get the global position of this instance
    int global_x = get_global_id(0);
    int global_y = get_global_id(1);

    // Load the network width into private memory
    int width = *network_width;
   
    // Multiply the iput values onto their row
    weights_vec_access(global_y,global_x) = weights_vec_access(global_y,global_x) * input_vec[global_y];

    // Find the local id of this instance in its workgroup
    uint local_id = get_local_id(0);

    // Get the work group size
    uint group_size = get_local_size(0);

    // Copy the numbers we want to use from global memory to local memory
    localSums[local_id] = weights_vec_access(global_y,global_x);

  // Loop for computing localSums
    for (uint stride = group_size/2; stride>0; stride /=2)
         {
         // Waiting for each 2x2 addition into given workgroup
         barrier(CLK_LOCAL_MEM_FENCE);

          // Divide WorkGroup into 2 parts and add elements 2 by 2
          // between local_id and local_id + stride
          if (local_id < stride)
            localSums[local_id] += localSums[local_id + stride];
         }


         bool multi_work_groups = true;

         if (multi_work_groups) {

               // Do this if we dont have multiple workgroups per row
            // Write result into partialSums[nWorkGroups]
            if (local_id == 0) {
              vec_out[global_y] = localSums[0];
            }

        } else{

      // Wait for all workgroups to get here
      barrier(CLK_GLOBAL_MEM_FENCE);
      // If the workgroup does not contain the row for the sum save it into global memory so we can do another summation
      
        }
 }                  

