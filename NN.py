from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import os



# Debug properties
DEBUG = True

# OpenCL properties
target_opencl_device_type = cl.device_type.GPU
opencl_device_list = []

# Network size properties
input_size = 3
hidden_size = 3
output_size = 3

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
	context = cl.Context(devices = opencl_device_list)
	return context

# Load an opencl kenrel file as a string
def cl_load_kernel(name):
	kernel = open(name,'r').read() 
	return kernel

# Multiply 2 vectors with opencl
def cl_mult_2_vec(this_context,input_vec_1,input_vec_2):

	queue = cl.CommandQueue(this_context)

	vec_1_to_device = cl_array.to_device(queue, input_vec_1)
	vec_2_to_device = cl_array.to_device(queue,input_vec_2)
	output_to_device = cl_array.empty_like(vec_1_to_device,queue)

	program = cl.Program(context,cl_load_kernel('component_multiply.cl')).build()

	program.component_multiply(queue, input_vec_1.shape, None, vec_1_to_device.data, vec_2_to_device.data, output_to_device.data)

	output_vec = output_to_device.get()

	return output_vec

# Add 2 vectors with opencl
def cl_add_2_vec(this_context,input_vec_1,input_vec_2):

	queue = cl.CommandQueue(this_context)

	vec_1_to_device = cl_array.to_device(queue, input_vec_1)
	vec_2_to_device = cl_array.to_device(queue,input_vec_2)
	output_to_device = cl_array.empty_like(vec_1_to_device,queue)

	program = cl.Program(context,cl_load_kernel('sum.cl')).build()

	program.sum(queue, input_vec_1.shape, None, vec_1_to_device.data, vec_2_to_device.data, output_to_device.data)

	output_vec = output_to_device.get()

	return output_vec

# Sum togther the elements of a vector
def cl_sum_vec(vec):

	queue = cl.CommandQueue(context)
	out = cl_array.sum(vec, dtype=np.ndarray, queue=queue, slice=None)

	return out

# Propigate values throught the network
def forward_prop():



	return 0

# Checks to see if a neuron meets its threshold to fire
def neuron_fire_check(val):
	# Trigger function
	out = np.tanh(val)
	if out >= neuron_fire_thresh:
		return True
	else
		return False




load_input_data('random')
init_data_structure()
cl_find_devices()
context = cl_get_context()

a = np.array([1,2,3,4])

out = cl_sum_vec(a)
print(out)















