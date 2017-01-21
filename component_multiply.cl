__kernel void component_multiply(
    __global const float *buffer_input, __global const float *buffer_weights, __global float *output_buffer)
{
  int id = get_global_id(0);
  output_buffer[id] = buffer_input[id] * buffer_weights[id];
}