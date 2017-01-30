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

def cl_load_debug(queue,local_group_size):
	debug_to_device = cl_array.empty(queue,local_group_size,dtype=np.int)
	return debug_to_device

# Propigate values through the network using a single kernel 
def feed_forward():
	# Make network_output global so we can write to it
	global network_output

	max_local_group_size = 400.0

	size_of_row = network_hidden.shape[1]

	num_groups_row = int(size_of_row/max_local_group_size)+1

	# Create a command queue
	queue = cl.CommandQueue(context)

	debug_to_device = cl_load_debug(queue,network_hidden.shape[1])

	# Move Network to device and return its pointers
	network_input_to_device,network_hidden_to_device,network_output_to_device = cl_move_network_to_device(queue)
	# Move hidden weights length to device
	hidden_width_to_device = cl_array.to_device(queue,network_hidden.shape[1]*np.ones(1).astype(np.int))
	# Move the number of work groups per row to device
	local_groups_per_row_to_device = cl_array.to_device(queue,num_groups_row*np.ones(1).astype(np.int))
	# Move a buffer to do sumations into the local memory of all workgroups 
	summ_local_to_device = cl.LocalMemory(sys.getsizeof(network_hidden[1,:]))

	# Specify the global and local work size
	global_work_size = network_hidden.shape

	# Unused at the moment but will be implemented later
	#pref_wrk_gSize = cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE

	# TODO: If collum size is bigger then max workgroup size then we need to take care of that
	local_work_size = (network_hidden.shape[1],1)

	# Build program
	program = cl.Program(context,cl_load_kernel('feed_forward.c')).build()

	# Call the kernel and load arguments
	program.feed_forward(queue,global_work_size, local_work_size, hidden_width_to_device.data,local_groups_per_row_to_device.data,debug_to_device.data,network_input_to_device.data , network_hidden_to_device.data,network_output_to_device.data,summ_local_to_device)

	debug_data = debug_to_device.get()

	print(debug_data)

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

print(network_input)
print(network_hidden)
print(output)
















