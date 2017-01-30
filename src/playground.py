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

def feed_forward_play():
	# Make network_output global so we can write to it
	global network_output

	# Create a command queue
	queue = cl.CommandQueue(context)

	# Move data to device and create a pointer to it.
	hidden_width_to_device = cl_array.to_device(queue,network_hidden.shape[1]*np.ones(1).astype(np.int))
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
	program = cl.Program(context,cl_load_kernel('feed_forward_play.c')).build()

	# Call the kernel and load arguments
	program.feed_forward_play(queue,global_work_size, local_work_size, hidden_width_to_device.data,network_input_to_device.data , network_hidden_to_device.data,network_output_to_device.data,summ_local_to_device)

	# Get the output from the device
	network_output = network_output_to_device.get()



network_hidden = 2*np.ones((8,8)).astype(np.float32)
network_input = np.array([1,2,3,4,5,6,7,8]).astype(np.float32)
network_output = np.ones(8).astype(np.float32)

for i in range(0,network_hidden.shape[0]):
	network_hidden[i,:] = i

cl_find_devices()
context = cl_get_context()
feed_forward_play()
print('Input vals: \n' + str(network_input))
print('hidden vals: \n' + str(network_hidden))
print('output vals: \n' + str(network_output))









