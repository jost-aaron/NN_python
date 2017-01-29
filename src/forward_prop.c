#define vec_out_access(r, c) (vec_out[(r)*width + (c)])
#define weights_vec_access(r, c) (weights_vec[(r)*width + (c)])

__kernel void forward_prop(
	__global int *network_width,
	__global int *network_height,
    __global float *input_vec,
    __global float *weights_vec, 
    __global float *vec_out,
    __local float *localSums)
{
  
    

    int global_x = get_global_id(0);
    int global_y = get_global_id(1);

    int width = *network_width;

   
    weights_vec_access(global_y,global_x) = weights_vec_access(global_y,global_x) * input_vec[global_y];


    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);

  // Copy from global memory to local memory
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

      // Write result into partialSums[nWorkGroups]
      if (local_id == 0)
        vec_out[global_y] = localSums[0];
 }                  

