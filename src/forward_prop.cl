__kernel void forward_prop(
    __global const float *input_vec, __global const float *weights_vec, __global float *vec_out)
{
  // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);
 
    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);




  
}