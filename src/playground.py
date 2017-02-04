import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import math
import os
import sys


target_cl_device_type = cl.device_type.GPU
cl_device_list = []
cl_device_work_group_max_size = []

os.environ["PYOPENCL_COMPILER_OUTPUT"] = '1'

def cl_find_devices():
	global cl_device_list
	plats = cl.get_platforms()
	for plat in plats:
		index = plats.index(plat)
		devices = plats[index].get_devices(target_cl_device_type)
		for device in devices: 
			cl_device_list.append(device)
			cl_device_work_group_max_size.append(device.max_work_group_size)

def cl_get_context():
	#context = cl.Context(devices = cl_device_list)
	context = cl.Context(cl_device_list)
	return context

# Load an opencl kenrel file as a string
def cl_load_kernel(name):
	kernel = open(name,'r').read() 
	return kernel

def estimate_vram_usage():

	# Number of values for hidden matrix
	num_vals = network_hidden.shape[0] * network_hidden.shape[1]
	
	# Number of values for input vector
	num_vals = num_vals + max(network_input.shape)
	
	# Number of values for output vector
	num_vals = num_vals + max(network_output.shape)
	
	# Number of values for the local buffer 
	num_vals = num_vals + max(network_hidden[0,:].shape)

	# Size of data in bytes
	in_bytes = num_vals*32/8

	# Size of data in kBytes
	in_Kbytes = in_bytes/1000

	# Size of data in MBytes
	in_Mbytes = in_Kbytes/1000

	# Check to see what the most relivent size is
	if (in_bytes < 1000):
		return str(in_bytes) + ' B'
	elif(in_Kbytes < 1000):
		return str(in_Kbytes) + ' kB'
	else:
		return str(in_Mbytes) + 'MB'

def verify_feed_forward():
	global DEBUG_network_output_multiplication
	global DEBUG_sum_bridge_after
	global network_output

	row_to_check = 0

	if (DEBUG_network_output_multiplication.shape[1] <= 256):
		for i in range(0,DEBUG_network_output_multiplication.shape[0]):
			row = DEBUG_network_output_multiplication[i,:]
			row_sum = sum(row)
			if (network_output[i] != row_sum):
				print('Errors in calculation')
				exit(1)
		print('Calculation Verifyed!')

	else:


		# Verify data
		found_error = 0
		found_another_error = 0
		for i in range(0,DEBUG_network_output_multiplication.shape[0]):
			row = DEBUG_network_output_multiplication[i,:]
			sum_bridge_result = DEBUG_sum_bridge_after[i,:]
			sum_bridge_sum = sum(sum_bridge_result)
			true_result = sum(row)

			if (found_error == 0 and found_another_error == 0 and true_result != sum_bridge_sum):
				found_error = 1
			elif (found_error == 1 and found_another_error >= 0 and true_result != sum_bridge_sum):
				found_another_error = found_another_error +1
			


			if (true_result != sum_bridge_sum and found_another_error == 0):
				row_to_check = i
				print('Row ' +str(row_to_check)+ ' of output: \n' + str(DEBUG_network_output_multiplication[row_to_check,:]))
				print('Row ' + str(row_to_check) + ' sum: ' + str(sum(DEBUG_network_output_multiplication[row_to_check,:])))
				print('Values in sum bride: \n' + str(DEBUG_sum_bridge_after[row_to_check,:]))
				print('Sum of row ' + str(row_to_check) + ' of sum bridge: ' + str(sum(DEBUG_sum_bridge_after[row_to_check,:])))
				print('Sum should be: ' + str(sum(DEBUG_network_output_multiplication[row_to_check,:])))
				print('Difference in sums: ' + str(sum(DEBUG_network_output_multiplication[row_to_check,:]) - sum(DEBUG_sum_bridge_after[row_to_check,:])))
				found_error = 1

			
		if (found_error == 0):
			print('Calculation Verifyed!')
		else:
			print(str(found_another_error) +  ' more rows have errors in calculations!')

def feed_forward_play():
	# Make network_output global so we can write to it
	global network_hidden
	global network_output
	global DEBUG_network_output_multiplication
	global DEBUG_sum_bridge_after

	# Create a command queue
	queue = cl.CommandQueue(context)

	# Find max local work group size
	max_work_group_size = min(cl_device_work_group_max_size)

	# calculte the number of work groups per row
	num_work_groups_per_row = int(network_hidden.shape[1]/max_work_group_size)+1

	# Initalize a variable for padding
	remainder_padding = 0

	# Make a temp variable we can add the padding to and not affect the network variable
	padded_matrix_tmp = network_hidden

	# If there was no padding added and there is only one work group per row
	if (num_work_groups_per_row != 1):

		# Calculate how much padding is necessary to fill the last work group
		remainder_padding = abs(max_work_group_size*(num_work_groups_per_row) - network_hidden.shape[1])

		# Generate a matrix with same number of rows as hidden network and number of collums defined by remainder_padding
		insert_padding = np.zeros((network_hidden.shape[0],remainder_padding)).astype(np.float32)

		# Make a temporary version of network_hidden and add padding (zeros) so the last workgroup is filled
		padded_matrix_tmp = np.append(padded_matrix_tmp,insert_padding,axis=1)
		
		# Check if the zeros were added to the last work group correctly
		average_work_group_size_test = padded_matrix_tmp.shape[1]/(1.0*num_work_groups_per_row)

		if (int(average_work_group_size_test) != 256):
			print('ERROR: Padding might now have been allocated correctly to fill up the last work group')
			exit(1)

	# create a buffer to store the sub sums for each row
	sum_bridge = np.zeros((network_hidden.shape[0],num_work_groups_per_row)).astype(np.float32)

	# move the buffer to the device
	sum_bridge_to_device = cl_array.to_device(queue,sum_bridge.flatten())

	# Move the number of sums per row
	sums_per_row_to_device = cl_array.to_device(queue,sum_bridge.shape[1]*np.ones(1).astype(np.int))

	#local_work_size = (network_hidden.shape[1]/num_work_groups_per_row,1)
	local_work_size = (0,0)
	if (padded_matrix_tmp.shape[1] <= 256):
		local_work_size = (padded_matrix_tmp.shape[1],1)
	else:
		local_work_size = (256,1)

	# Move data to device and create a pointer to it.
	hidden_width_to_device = cl_array.to_device(queue,padded_matrix_tmp.shape[1]*np.ones(1).astype(np.int))
	network_hidden_to_device = cl_array.to_device(queue, padded_matrix_tmp.flatten())
	network_input_to_device = cl_array.to_device(queue, network_input)
	network_output_to_device = cl_array.empty(queue, len(network_output), dtype=np.float32)
	sum_local_to_device = cl.LocalMemory(sys.getsizeof(padded_matrix_tmp[1,:]))

	# Specify the global and local work size
	global_work_size = (padded_matrix_tmp.shape[1],padded_matrix_tmp.shape[0])

	# Build program
	program = cl.Program(context,cl_load_kernel('feed_forward_play.c')).build()

	# Call the kernel and load arguments
	program.feed_forward_play(queue,
								global_work_size, 
								local_work_size, 
								hidden_width_to_device.data,
								sums_per_row_to_device.data,
								network_input_to_device.data , 
								network_hidden_to_device.data,
								network_output_to_device.data,
								sum_bridge_to_device.data,
								sum_local_to_device)

	# Get the output from the device
	network_output = network_output_to_device.get()
	DEBUG_network_output_multiplication = network_hidden_to_device.get()
	DEBUG_network_output_multiplication = np.resize(DEBUG_network_output_multiplication,(padded_matrix_tmp.shape[0],padded_matrix_tmp.shape[1]))
	DEBUG_sum_bridge_after  = sum_bridge_to_device.get()
	DEBUG_sum_bridge_after.resize(sum_bridge.shape)
	



data_size = 256+100
print('Generating hidden data...')
network_hidden = np.ones((data_size,data_size)).astype(np.float32)

array_vals = []
print('Generating input data...')
for i in range(1,data_size+1):
	array_vals.append(i)

network_input = np.array(array_vals).astype(np.float32)
print('Generating output buffer...')
network_output = np.ones(network_hidden.shape[0]).astype(np.float32)

# Debug place to store hidden values after they have been multiplied
DEBUG_network_output_multiplication = 0

# Edit the values of the initalized weights matrix so there is enough variablilty to debug
for i in range(0,network_hidden.shape[0]):
	network_hidden[i,:] = i

cl_find_devices()
context = cl_get_context()
print('Propigating values through network...')
feed_forward_play()



print('WARNING: Opperation is very slow: \n     VRAM usage estimate: ' + estimate_vram_usage())


# DEBUG OUTPUT STUFF
print('Verifying output...')
verify_feed_forward()





