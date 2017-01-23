from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import os
import time



# Debug properties
DEBUG = True

# OpenCL properties
target_opencl_device_type = cl.device_type.GPU
opencl_device_list = []

# Network size properties
input_size = 500
hidden_size = 500
output_size = 500

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
	global opencl_device_list
	plats = cl.get_platforms()
	for plat in plats:
		index = plats.index(plat)
		devices = plats[index].get_devices(target_opencl_device_type)
		for device in devices: 
			opencl_device_list.append(device)
			

	print('Number of OpenCl devices found: ' + str(len(opencl_device_list)))

# Get the context for a given device
def cl_get_context():
	#context = cl.Context(devices = opencl_device_list)
	context = cl.Context(opencl_device_list)
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

	program = cl.Program(context,cl_load_kernel('component_multiply.cl')).build()

	program.component_multiply(queue, input_vec_1.shape, None, vec_1_to_device.data, vec_2_to_device.data, output_to_device.data)

	output_vec = output_to_device.get()

	return output_vec

# Add 2 vectors with opencl
def cl_add_2_vec(input_vec_1,input_vec_2):

	queue = cl.CommandQueue(context)

	vec_1_to_device = cl_array.to_device(queue, input_vec_1)
	vec_2_to_device = cl_array.to_device(queue,input_vec_2)
	output_to_device = cl_array.empty_like(vec_1_to_device,queue)

	program = cl.Program(context,cl_load_kernel('sum.cl')).build()

	program.sum(queue, input_vec_1.shape, None, vec_1_to_device.data, vec_2_to_device.data, output_to_device.data)

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


# Propigate values throught the network
def forward_prop():

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
forward_prop()
#print(network_output)















