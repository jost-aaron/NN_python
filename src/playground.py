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

	size = (1.0*network_hidden.shape[0]*network_hidden.shape[0])/min(cl_device_work_group_max_size)
	size = math.sqrt(size)
	if size != int(size):
		size = int(size)

	#global_size = (size+1,size+1)
	#local_size = (int(math.sqrt(min(cl_device_work_group_max_size))),int(math.sqrt(min(cl_device_work_group_max_size))))


	global_size = (1,network_hidden.shape[0]**2)
	local_size = None

	global_size_to_device = cl_array.to_device(queue,np.int32(global_size[1]))
	
	weights_matrix_to_device = cl_array.to_device(queue, network_hidden)
	input_vec_to_device = cl_array.to_device(queue,network_input)

	output_to_device = cl_array.empty_like(weights_matrix_to_device,queue)

	program = cl.Program(context,cl_load_kernel('forward_prop.c')).build()

	program.forward_prop(queue, global_size, (1,1), global_size_to_device.data,weights_matrix_to_device.data, input_vec_to_device.data, output_to_device.data)
	output_vec = output_to_device.get()
	return output_vec


network_hidden = np.ones((5*5)).astype(np.float32)
network_input = 5*np.ones(5).astype(np.float32)


cl_find_devices()
context = cl_get_context()
out = forward_prop()
print('Input vals: ' + str(network_input))
print('hidden vals: ' + str(network_hidden))
print('output vals: ' + str(out))
