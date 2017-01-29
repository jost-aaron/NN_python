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
input_size = 8
hidden_size = 8
output_size = 8

# Neuron Properties
neuron_fire_thresh = 0.5

# Network storage variables
# Data structure for hidden, each Row is a hidden neuron
network_input = []
network_hidden = []
network_output = []

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
			

	print('Number of OpenCl devices found: ' + str(len(cl_device_list)))

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

# Sum togther the elements of a vector
def cl_sum_bad_vec(vec):

	# Split up the input vector into 2 seperate ones to sum
	length = len(vec)
	a = vec[0:(length/2)]
	b = vec[length/2:length]
	length_a = len(a)
	length_b = len(b)

	# Check if the vectors are equal in length and if not add vec(end) of the larger vector to vec(end-1)
	if (length_a > length_b):

		a_end = a[length_a]
		a = a[0:length_a-1]
		a[len(a)] = a[len(a)] + a_end

	elif (length_b > length_a):

		b_end = b[length_b-1]
		b = b[0:length_b-1]
		b[len(b)-1] = b[len(b)-1] + b_end

	# If the vectors were not correctly sized crash the program
	if (len(a) != len(b)):
		print('Error tried to sum to vectors with different lengths in: cl_sum_bad_vec')
		exit(1)

	current = cl_add_2_vec(a,b)

	if len(current) > 25:
		return cl_sum_bad_vec(current)
	else:
		return sum(current)

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


# Propigate values throught the network
def forward_prop_bad():

	start_time = time.time()
	done = 0.0

	for i in range(0,hidden_size):
		mult = cl_mult_2_vec(network_hidden[i,:],network_input)
		sum_mine = cl_sum_bad_vec(mult)
		network_output[i]=neuron_fire_check(sum_mine)
		
		print("Percent done: " + str(i) + "/" + str(hidden_size))


	elapsed_time = time.time() - start_time
	print('OpenCl Time: ' + str(elapsed_time))
		
	start_time = time.time()
	for i in range(0,hidden_size):
		network_output[i]=neuron_fire_check(sum(network_hidden[i,:]*network_input))
	elapsed_time = time.time() - start_time
	print('CPU Time: ' + str(elapsed_time))

	return 0

def forward_prop():
	# Make network_output global so we can write to it
	global network_output

	# Create a command queue
	queue = cl.CommandQueue(context)

	# Move data to device and create a pointer to it.
	hidden_width_to_device = cl_array.to_device(queue,network_hidden.shape[1]*np.ones(1).astype(np.int))
	hidden_height_to_device = cl_array.to_device(queue,network_hidden.shape[0]*np.ones(1).astype(np.int))
	network_hidden_to_device = cl_array.to_device(queue, network_hidden.flatten('F'))
	network_input_to_device = cl_array.to_device(queue, network_input)
	network_output_to_device = cl_array.empty(queue, len(network_output), dtype=np.float32)
	summ_local_to_device = cl.LocalMemory(sys.getsizeof(network_hidden[1,:]))

	# Specify the global and local work size
	global_work_size = network_hidden.shape

	# Unused at the moment but will be implemented later
	#pref_wrk_gSize = cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE

	# TODO: If collum size is bigger then max workgroup size then we need to take care of that
	local_work_size = (network_hidden.shape[1],1)

	# Build program
	program = cl.Program(context,cl_load_kernel('forward_prop.c')).build()

	# Call the kernel and load arguments
	program.forward_prop(queue,global_work_size, local_work_size, hidden_width_to_device.data, hidden_height_to_device.data,network_input_to_device.data , network_hidden_to_device.data,network_output_to_device.data,summ_local_to_device)

	# Get the output from the device
	return network_output_to_device.get()



# Checks to see if a neuron meets its threshold to fire
def neuron_fire_check(val):
	# Trigger function
	out = np.tanh(val)
	if out >= neuron_fire_thresh:
		return 1
	else:
		return 0




load_input_data('random')
init_data_structure()
cl_find_devices()
context = cl_get_context()
forward_prop_bad()
output = forward_prop()

print(network_input)
print(network_hidden)
print(output)
















