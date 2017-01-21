from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import os



DEBUG = True
target_opencl_device_type = cl.device_type.GPU


# Network size properties
input_size = 3
hidden_size = 3
output_size = 3

# Network storage variables
# Data structure for hidden, each Row is a hidden neuron
network_input = []
network_hidden = []
network_output = []

# OpenCl Device list
opencl_device_list = []


def debug_output(message):
	if DEBUG:
		print(message)

def init_data_structure():
	global network_hidden
	global network_output
	network_hidden = np.ones((hidden_size,input_size)).astype(np.float32)
	network_output = (np.zeros(output_size)).astype(np.float32)

def load_input_data(data_type):
	global network_input
	if data_type == 'random':
		#network_input = np.random.rand(input_size) + 10*np.ones(input_size)
		network_input = 10*np.ones(input_size).astype(np.float32)
		network_input[1]=7
		network_input[2]=8
	elif data_type == 'from':
		return 0

def cl_find_devices():
	global opencl_device_list
	plats = cl.get_platforms()
	for plat in plats:
		index = plats.index(plat)
		devices = plats[index].get_devices(target_opencl_device_type)
		for device in devices: 
			opencl_device_list.append(device)

	print('Number of OpenCl devices found: ' + str(len(opencl_device_list)))

def cl_get_context():
	context = cl.Context(devices = opencl_device_list)
	return context

def cl_load_kernel(name):
	kernel = open(name,'r').read() 
	return kernel


def cl_mult_2_vec(this_context,input_vec_1,input_vec_2):

	queue = cl.CommandQueue(this_context)

	vec_1_to_device = cl_array.to_device(queue, input_vec_1)
	vec_2_to_device = cl_array.to_device(queue,input_vec_2)
	output_to_device = cl_array.empty_like(vec_1_to_device,queue)

	program = cl.Program(context,cl_load_kernel('component_multiply.cl')).build()

	program.component_multiply(queue, input_vec_1.shape, None, vec_1_to_device.data, vec_2_to_device.data, output_to_device.data)

	output_vec = output_to_device.get()

	return output_vec

def cl_add_2_vec(this_context,input_vec_1,input_vec_2):

	queue = cl.CommandQueue(this_context)

	vec_1_to_device = cl_array.to_device(queue, input_vec_1)
	vec_2_to_device = cl_array.to_device(queue,input_vec_2)
	output_to_device = cl_array.empty_like(vec_1_to_device,queue)

	program = cl.Program(context,cl_load_kernel('sum.cl')).build()

	program.sum(queue, input_vec_1.shape, None, vec_1_to_device.data, vec_2_to_device.data, output_to_device.data)

	output_vec = output_to_device.get()

	return output_vec

def cl_sum_vec(vec):

	queue = cl.CommandQueue(context)
	out = cl_array.sum(vec, dtype=np.ndarray, queue=queue, slice=None)

	return out

def forward_prop():



	return 0


load_input_data('random')
init_data_structure()
cl_find_devices()
context = cl_get_context()

a = np.array([1,2,3,4])

out = cl_sum_vec(a)
print(out)















