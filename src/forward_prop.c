__kernel void forward_prop(
     __global const int *work_group_max_size, __global const float *input_vec, __global const float *weights_vec, __global float *vec_out)
{
  // Work group index in global group
    int work_group_x = get_group_id(0);
    int work_group_y = get_group_id(1);
 
    // work item index in workgroup
    int work_item_x = get_local_id(0);
    int work_item_y = get_local_id(1);





  
}