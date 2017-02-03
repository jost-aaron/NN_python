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

def estimate_vram_usage(num_vals):
	in_bytes = num_vals*32/8
	in_Kbytes = in_bytes/1000
	in_Mbytes = in_Kbytes/1000

	if (in_bytes < 1000):
		return str(in_bytes) + ' B'
	elif(in_Kbytes < 1000):
		return str(in_Kbytes) + ' kB'
	else:
		return str(in_Mbytes) + 'MB'


def feed_forward_play():
	# Make network_output global so we can write to it
	global network_output
	global network_output_multiplication

	# Create a command queue
	queue = cl.CommandQueue(context)

	# Move data to device and create a pointer to it.
	hidden_width_to_device = cl_array.to_device(queue,network_hidden.shape[1]*np.ones(1).astype(np.int))
	network_hidden_to_device = cl_array.to_device(queue, network_hidden.flatten())
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
	network_output_multiplication = network_hidden_to_device.get()
	network_output_multiplication.resize((data_size,data_size))



data_size = 210
network_hidden = np.ones((data_size,data_size)).astype(np.float32)

array_vals = []
for i in range(1,data_size+1):
	array_vals.append(i)

network_input = np.array(array_vals).astype(np.float32)
network_output = np.ones(data_size).astype(np.float32)

# Debug place to store hidden values after they have been multiplied
network_output_multiplication = 0

# Edit the values of the initalized weights matrix so there is enough variablilty to debug
for i in range(0,network_hidden.shape[0]):
	network_hidden[i,:] = i

cl_find_devices()
context = cl_get_context()
feed_forward_play()

# For calculating the expected value of the output computation
expected_out = np.zeros(network_hidden.shape[1])
for i in range(0,network_hidden.shape[1]):
	expected_out[i] = sum(network_output_multiplication[i,:])

global_vram_usage = estimate_vram_usage(network_hidden.shape[0]*network_hidden.shape[1] + max(network_input.shape) + max(network_output.shape))
print('Global VRAM usage: ' + global_vram_usage)
print('Local VRAM usage: ' + estimate_vram_usage(max(network_hidden[1,:].shape)))

#print('Input vals: \n' + str(network_input))
#print('hidden vals: \n' + str(network_hidden))
#print('Hidden After mult: \n' + str(network_output_multiplication))
#print('output vals: \n' + str(network_output))
#print('Expected output: \n' + str(expected_out))

diff = network_output - expected_out

if (sum(diff) == 0):
	print('Addition Sucessfull!')
else:
	print('Addition Unsucessfull')
	print('Difference: \n' + str(diff))









