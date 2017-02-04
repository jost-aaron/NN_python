from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import os
import time
import sys



# Debug properties
DEBUG = True

# OpenCL properties
target_cl_device_type = cl.device_type.GPU
cl_device_list = []
cl_device_work_group_max_size = []

# Network size properties
input_size = 8000
hidden_size = 8000
output_size = 8000

# Neuron Properties
neuron_fire_thresh = 0.5

# Network storage variables
# Data structure for hidden, each Row is a hidden neuron
network_input = []
network_hidden = []
network_output = []

# Timer Class for having instances of a timer for benchmarking
class Timer:

	def __init__(self):
		self.t = 0

	def start(self):
		self.t = time.time()

	def get_elapsed(self):
		return time.time() - self.t
	
	def print_elapsed_time(self):
		print('Time Elapsed: ' + str(time.time() - self.t))

	def print_elapsed_time_msg(self,msg):
		print(msg + str(time.time() - self.t))

	def reset(self):
		self.t = time.time()

# Easy way to output dubug information
def debug_output(message):
	if DEBUG:
		print(message)

# Create the hidden and output data structure with numpy arrays
def init_data_structure():
	global network_hidden
	global network_output
	network_hidden = np.ones((hidden_size,input_size)).astype(np.float32)
	network_output = (np.zeros(output_size)).astype(np.float32)

# Load some input data to feed to the network
def load_input_data(data_type):
	global network_input
	if data_type == 'random':
		network_input = 10*np.ones(input_size).astype(np.float32)
	elif data_type == 'from':
		return 0

# Find and cataloge all of the opencl compatable devices on the system
def cl_find_devices():
	global cl_device_list
	plats = cl.get_platforms()
	for plat in plats:
		index = plats.index(plat)
		devices = plats[index].get_devices(target_cl_device_type)
		for device in devices: 
			cl_device_list.append(device)
			cl_device_work_group_max_size.append(device.max_work_group_size)
			
	print('==========================================================')
	print('=======      OpenCL Devices on this platform      ========')
	print('==========================================================')
	print('Number of OpenCl devices found: ' + str(len(cl_device_list)))
	for device in cl_device_list:
		cl_print_device_information(device)
	print('==========================================================')
	print('=======            Current Computation            ========')
	print('==========================================================')

# Get the context for a given device
def cl_get_context():
	#context = cl.Context(devices = cl_device_list)
	context = cl.Context(cl_device_list)
	return context

# Load an opencl kenrel file as a string
def cl_load_kernel(name):
	kernel = open(name,'r').read() 
	return kernel

# Multiply 2 vectors with opencl
def cl_mult_2_vec(input_vec_1,input_vec_2):

	queue = cl.CommandQueue(context)

	vec_1_to_device = cl_array.to_device(queue, input_vec_1)
	vec_2_to_device = cl_array.to_device(queue,input_vec_2)
	output_to_device = cl_array.empty_like(vec_1_to_device,queue)

	program = cl.Program(context,cl_load_kernel('component_multiply.c')).build()

	program.component_multiply(queue, input_vec_1.shape, None, vec_1_to_device.data, vec_2_to_device.data, output_to_device.data)

	output_vec = output_to_device.get()

	return output_vec

# Add 2 vectors with opencl
def cl_add_2_vec(input_vec_1,input_vec_2):

	queue = cl.CommandQueue(context)

	vec_1_to_device = cl_array.to_device(queue, input_vec_1)
	vec_2_to_device = cl_array.to_device(queue,input_vec_2)
	output_to_device = cl_array.empty_like(vec_1_to_device,queue)

	program = cl.Program(context,cl_load_kernel('component_sum.c')).build()

	program.component_sum(queue, input_vec_1.shape, None, vec_1_to_device.data, vec_2_to_device.data, output_to_device.data)

	output_vec = output_to_device.get()

	return output_vec

def cl_print_device_information(device):
	print("----------------------------------------------------------")
	print("Device name:", device.name)
	print("Device type:", cl.device_type.to_string(device.type))
	print("Device memory: ", device.global_mem_size//1024//1024, 'MB')
	print("Device max clock speed:", device.max_clock_frequency, 'MHz')
	print("Device compute units:", device.max_compute_units)
	print("Device max work group size:", device.max_work_group_size)
	print("Device max work item sizes:", device.max_work_item_sizes)
	print("----------------------------------------------------------")

def cl_move_network_to_device(queue):
	network_hidden_to_device = cl_array.to_device(queue, network_hidden.flatten('F'))
	network_input_to_device = cl_array.to_device(queue, network_input)
	network_output_to_device = cl_array.empty(queue, len(network_output), dtype=np.float32)

	return [network_input_to_device,network_hidden_to_device,network_output_to_device]

# Propigate values through the network using a single kernel 
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

def feed_forward():
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
	program = cl.Program(context,cl_load_kernel('feed_forward.c')).build()

	# Call the kernel and load arguments
	program.feed_forward(queue,
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

# Checks to see if a neuron meets its threshold to fire
def neuron_fire_check(val):
	# Trigger function
	out = np.tanh(val)
	if out >= neuron_fire_thresh:
		return 1
	else:
		return 0



DEBUG_network_output_multiplication = 0
DEBUG_sum_bridge_after = 0


t = Timer()


load_input_data('random')
init_data_structure()
cl_find_devices()
context = cl_get_context()


for i in range(0,network_hidden.shape[0]):
	network_hidden[i,:] = i



t.start()

output = feed_forward()

t.print_elapsed_time()
verify_feed_forward()

















