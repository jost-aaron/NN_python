
// Documentation of kernel

//   data structure of network this kernel calculates is as follows:
//   -this example:
//   3 input neurons
//   2 hidden neurons
//   4 output neurons

//          First instance of kernel                            Second instance of kernel
// |-----------------------------------------|        |------------------------------------------|
//   
//   Input vector:          Hidden weights:           Result from instance 1:     Output Weights:    Output vector:
//   
//   | Input_1 |          |   w1   w2  w3  |   -->      |  intermediate_1  |        | w1  w2 |  -->  | Output_1 |
//   | Input_2 |    OPP   |   w1   w2  w3  |   -->      |  intermediate_2  |  OPP   | w1  w2 |  -->  | Output_2 |
//   | Input_3 |                                                                    | w1  w2 |  -->  | Output_3 |
//                                                                                  | w1  w2 |  -->  | Output_4 |

// OPP Definition:
//   The kernel will multiply each collum of the weight matrix by the input value at the same index as the weights collum.
//   Note: w1 corrisponds to the a input neuron at the index of its number and each w1 and w2 is independant even though they have the same name


// Defined functions to access the flattened matricies as regular matricies
#define input_matrix_access(r, c) (input_matrix[(r)*width + (c)])
#define sum_bridge(r,c) (sum_bridge[(r)*num_sums_per_collum + (c)])

__kernel void forward_prop(
    __global uint *network_width,
    __global uint *sums_per_collum,
    __global float *input_vector,
    __global float *input_matrix, 
    __global float *output_vector,
    __global float *sum_bridge,
    __local float *local_sums)
{
  
    // Get the global position of this instance
    uint global_x = get_global_id(0);
    uint global_y = get_global_id(1);

    // Load the constants into private memory
    uint width = *network_width;
    uint num_sums_per_collum = *sums_per_collum;
   
    // Multiply the iput values onto their row
    // For some reason this was transposing the matrix when accesing input_matrix_access so it was switched
    input_matrix_access(global_y,global_x) = input_matrix_access(global_y,global_x) * input_vector[global_x];

    // Find the local id of this instance in its workgroup
    uint local_id = get_local_id(0);

    // Get the work group size
    uint group_size = get_local_size(0);

    // Copy the numbers we want to use from global memory to local memory
    local_sums[local_id] = input_matrix_access(global_y,global_x);

  // Loop for computing local_sums of this row
    for (float stride = group_size/(float)2; (int)stride>0; stride /=2)
         {
         // Wait for the addition of the last stride to finish
         barrier(CLK_LOCAL_MEM_FENCE);

         // If stride is an odd number add the last value of the current sum to the value at local index 0 and make stride an even number
         if (!(stride - (float)((int)(stride)) == 0) && local_id == 0){
            local_sums[0] += local_sums[2*(uint)stride];
            stride = (float)((int)stride);
         }

         // Wait for the odd stride to be fixed
         barrier(CLK_LOCAL_MEM_FENCE);

          // Divide WorkGroup into 2 parts and add elements 2 by 2 between local_id and local_id + stride
          if (local_id < (uint)stride)
            local_sums[local_id] += local_sums[local_id + (uint)stride];

          barrier(CLK_LOCAL_MEM_FENCE);
         }

         barrier(CLK_LOCAL_MEM_FENCE);




         // Check if we need to sum values with another work group for this row
         if (num_sums_per_collum == 1) {

            // Write result into the output vector
            if (local_id == 0) {
              output_vector[global_y] = local_sums[0];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

        } else{

          // If the local id is 0 then move the sum of this workgroup into the sum bridge in the correct collum
          if (local_id == 0){
            
            // Calculate the collum we need to store this workgroups result in the sub bridge
            uint collum = (int)(global_x/group_size);

            // Move the result from local memory into global memory of sub bridge
            sum_bridge(global_y,collum) = local_sums[0];
          }


          // Wait for all workgroups to finish their local sums and put them into the sum bridge
         // barrier(CLK_GLOBAL_MEM_FENCE);

          // If this instance is the leader of its row
          //if (global_x == 0) {
            // Move number of sums per collum into local memory
          //  uint num_sums = *sums_per_collum;

            // ititalize a value for total number of sums
          //  float total_sum = 0;

            // Sum the sum bridge row
         //   for (uint i = 0; i < num_sums; i++) {
         //       total_sum += sum_bridge(global_y,i);
          //    }

            // Move the result into the output vector
         //   output_vector[global_y] = total_sum;

         // }
    

        }



 }                  