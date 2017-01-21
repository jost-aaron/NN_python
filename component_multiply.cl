__kernel void component_multiply(
    __global const float *vec_1, __global const float *vec_2, __global float *vec_out)
{
  int id = get_global_id(0);
  vec_out[id] = vec_1[id] * vec_2[id];
}