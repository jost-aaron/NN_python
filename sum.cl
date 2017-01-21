__kernel void sum(__global const float *vec_1, __global const float *vec_2, __global float *vec_out)
    {
      int gid = get_global_id(0);
      cvec_out[gid] = vec_1[gid] + vec_2[gid];
    }