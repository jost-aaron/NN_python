// Defined functions to access the flattened matricies as regular matricies
#define input_matrix_access(r, c) (matrix_in[(r)*width + (c)])

__kernel void sum_rows(
    __global float *matrix_in,
    __global uint *matrix_width,
    __global float *vector_out
    )
{
    // Get the global position of this instance
    uint global_x = get_global_id(0);
    uint global_y = get_global_id(1);

    uint local_id = get_local_id(0);
    
    uint width = *matrix_width;

    float local_sum = 0;

    if (local_id == 0) {
      for (int i = 0; i < width; i++) {
        local_sum += input_matrix_access(global_y,global_x+i);
      }
      vector_out[global_y] = local_sum;
    }
 }                  