#define vec_out_access(r, c) (vec_out[(r)*width + (c)])
#define weights_vec_access(r, c) (weights_vec[(r)*width + (c)])

__kernel void forward_prop(
	__global int *network_width,
	__global int *network_height,
    __global float *input_vec,
    __global float *weights_vec, 
    __global float *vec_out)
{
  
    int global_x = get_global_id(0);
    int global_y = get_global_id(1);
    int width = *network_width;

   
    weights_vec_access(global_y,global_x) = weights_vec_access(global_y,global_x) * input_vec[global_y];

    // Find max work group size 
    // allocate the work group sizes so they cover part of their row in the matrix
    // Each work group will cascade its sum

  
}