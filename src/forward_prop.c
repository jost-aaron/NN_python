__kernel void forward_prop(
     __global int *global_size,
     __global float *input_vec,
     __global float *weights_vec, 
     __global float *vec_out)
{
  // Work group index in global group
    int global_x = get_group_id(0);
    int global_y = get_group_id(1);



	vec_out[global_y] = input_vec[global_y] * weights_vec[global_y];

  
}