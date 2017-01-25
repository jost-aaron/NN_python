import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import math
import os


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

def forward_prop():


	queue = cl.CommandQueue(context)

	# Move data to device and give a pointer to it.
	hidden_width_to_device = cl_array.to_device(queue,network_hidden.shape[1]*np.ones(1).astype(np.int))
	network_hidden_to_device = cl_array.to_device(queue, network_hidden.flatten('F'))
	network_input_to_device = cl_array.to_device(queue, network_input)
	network_output_to_device = cl_array.empty_like(network_hidden_to_device,queue)

	# Specify the global and local work size
	global_work_size = network_hidden.shape
	number_work_groups = 4
	local_work_size = (network_hidden.shape[0]/number_work_groups,network_hidden.shape[1]/number_work_groups)

	# Build program
	program = cl.Program(context,cl_load_kernel('forward_prop.c')).build()

	# Call the kernel and load arguments
	program.forward_prop(queue, global_work_size, None, hidden_width_to_device.data, network_input_to_device.data , network_hidden_to_device.data,network_output_to_device.data)

	network_output = network_output_to_device.get().resize((8,8))



network_hidden = 3*np.ones((8,8)).astype(np.float32)
network_input = 2*np.ones(8).astype(np.float32)
network_output = np.ones((8,8)).astype(np.float32)



cl_find_devices()
context = cl_get_context()
forward_prop()
print('Input vals: ' + str(network_input))
print('hidden vals: ' + str(network_hidden))
print('output vals: ' + str(network_output))
