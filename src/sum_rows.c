// This file is used for the following things:
//   - Summing the rest of the values which could not be globaly syncronized in feed_forward.c 
//   - Applying the activation function to the output of the hidden and output neurons


// Defined functions to access the flattened matricies as regular matricies
#define input_matrix_access(r, c) (matrix_in[(r)*width + (c)])


float HTan(float val);
float Logistic(float val);

// Hyperbolic tangent funciton from -1 to 1
float HTan(float val){
    return (2/(1+exp(-2*val))) - 1;
}

// Logistic Equation from 0 to 1
float Logistic(float val){
    return 1/(1+exp(-val));
}


__kernel void sum_rows(
    __global float *matrix_in,
    __global uint *matrix_width,
    __global float *vector_out,
    __global uint *calc_fire_in,
    __global uint *activation_function_type_in)
{
    // Get the global position of this instance
    uint global_x = get_global_id(0);
    uint global_y = get_global_id(1);

    uint local_id = get_local_id(0);
    
    int width = *matrix_width;
    int calc_fire = *calc_fire_in;
    int activation_function_type = *activation_function_type_in;

    float local_sum = 0;

    if (local_id == 0) {
      for (int i = 0; i < width; i++) {
        local_sum += input_matrix_access(global_y,global_x+i);
      }

      if (calc_fire == 1){
        if (activation_function_type == 0){
            local_sum = HTan(local_sum);
        } else if(activation_function_type == 1) {
            local_sum = Logistic(local_sum);
        } 
      }
      vector_out[global_y] = local_sum;
    }
 }                  