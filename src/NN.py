from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import os
import time
import sys
import colorama
from colorama import Fore, Back, Style

# Clear the screen
os.system('cls')

# Initilize colorama to get console colors
colorama.init()


DEBUG = True

# OpenCL properties
target_cl_device_type = cl.device_type.GPU
cl_device_list = []
cl_device_work_group_max_size = []

# Network size properties
input_size = 700000
hidden_size = 800
output_size = 20

# Neuron Properties
neuron_fire_thresh = 0.5

# Network storage variables
# Data structure for hidden, each Row is a hidden neuron
network_input = []
network_hidden = []
network_output = []
network_output_weights = []

# Timer Class for having instances of a timer for benchmarking
class Timer:

	def __init__(self):
		self.t = 0

	def start(self):
		self.t = time.time()

	def get_elapsed(self):
		return time.time() - self.t
	
	def print_elapsed_time(self):

		seconds = time.time() - self.t
		minutes = 0
		while (seconds > 60):
				minutes = minutes + 1
				seconds = seconds - 60
		if (minutes == 0):
			time_elapsed = str(round(seconds,5)) + ' seconds'
		else:
			time_elapsed = str(minutes) + ' minutes ' + str(round(seconds,5)) + ' seconds '

		print(Back.BLUE+'Time Elapsed: ' +Back.YELLOW +Fore.BLACK+ ' '+ time_elapsed + ' '+ Style.RESET_ALL)

	def print_elapsed_time_msg(self,msg):
		print(msg + str(time.time() - self.t))

	def reset(self):
		self.t = time.time()

# Create the hidden and output data structure with numpy arrays
def init_data_structure():
	global network_hidden
	global network_output
	global network_output_weights
	network_hidden = np.ones((hidden_size,input_size)).astype(np.float32)
	network_output = (np.zeros(output_size)).astype(np.float32)
	network_output_weights = np.ones((output_size,hidden_size)).astype(np.float32)

# Load some input data to feed to the network
def load_input_data(data_type):
	global network_input
	if data_type == 'random':
		network_input = 10*np.ones(input_size).astype(np.float32)
	elif data_type == 'from':
		return 0

def print_color_fore(text,color):
	color = color.upper()

	if 	(color == 'GREEN'):
		print(Fore.GREEN + text + Style.RESET_ALL)
	elif(color == 'YELLOW'):
		print(Fore.YELLOW + text + Style.RESET_ALL)
	elif(color == 'BLUE'):
		print(Fore.BLUE + text + Style.RESET_ALL)
	elif(color == 'WHITE'):
		print(Fore.WHITE + text + Style.RESET_ALL)
	elif(color == 'RED'):
		print(Fore.RED + text + Style.RESET_ALL)
	elif(color == 'PURPLE'):
		print(Fore.PURPLE + text + Style.RESET_ALL)
	elif(color == 'BROWN'):
		print(Fore.BROWN + text + Style.RESET_ALL)
	elif(color == 'BLACK'):
		print(Fore.BLACK + text + Style.RESET_ALL)
	elif(color == 'GREEN'):
		print(Fore.GREEN + text + Style.RESET_ALL)

	print()

def print_network_information():
	# Len 59
	print(Fore.BLACK + Back.WHITE+'==========================================================')
	print('=======' + Back.GREEN + Fore.WHITE+'            Network Information            '+Back.WHITE+Fore.BLACK+'========')
	print('==========================================================' + Style.RESET_ALL)
	print(Fore.WHITE+Back.BLUE+'Input size: ' + Fore.BLACK+Back.YELLOW+ ' '+ str(format(input_size,',d'))+ ' ' +Style.RESET_ALL)
	print(Fore.WHITE+Back.BLUE+'Hidden size: ' + Fore.BLACK+Back.YELLOW+' '+str(format(hidden_size,',d'))+ ' '+ Fore.BLACK+Back.YELLOW+Style.RESET_ALL)
	print(Fore.WHITE+Back.BLUE+'Output size: ' + Fore.BLACK+Back.YELLOW+' '+str(format(output_size,',d'))+  ' ' +Style.RESET_ALL)
	print(Fore.WHITE+Back.BLUE+'Number of data points: '  +Fore.BLACK+Back.YELLOW+ ' '+str(format(input_size + input_size*hidden_size + output_size,',d'))+ ' '+Fore.BLACK+Back.YELLOW+ Style.RESET_ALL)
	print(Fore.WHITE+Back.BLUE+'Estimated on device data size: ' +Fore.BLACK+Back.YELLOW+ ' '+estimate_vram_usage() + ' '+Fore.BLACK+Back.YELLOW+ Style.RESET_ALL)
	print(Fore.BLACK + Back.WHITE+'==========================================================')
	print('======='+ Back.GREEN + Fore.WHITE+'            Current Computation            '+Back.WHITE+Fore.BLACK+'========')
	print('=========================================================='+ Style.RESET_ALL)

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
			
	print(Fore.BLACK + Back.WHITE+'==========================================================')
	print('======='+ Back.GREEN + Fore.WHITE+'      OpenCL Devices on this platform      '+Back.WHITE+Fore.BLACK+'========')
	print('=========================================================='+ Style.RESET_ALL)
	print(Back.BLUE + Fore.WHITE+'Number of OpenCl devices found: '+ Back.YELLOW + Fore.BLACK +' '+ str(len(cl_device_list)) + ' ' + Style.RESET_ALL)
	for device in cl_device_list:
		num = cl_device_list.index(device)
		cl_print_device_information(device,num)

# Get the context for a given device
def cl_get_context():
	#context = cl.Context(devices = cl_device_list)
	context = cl.Context(cl_device_list)
	return context

# Load an opencl kenrel file as a string
def cl_load_kernel(name):
	kernel = open(name,'r').read() 
	return kernel

# Print out information about a specific opencl device
def cl_print_device_information(device,number):
	if (number == 0):
		print(Fore.BLACK+Back.WHITE+"----------------------------------------------------------" + Style.RESET_ALL)
	print(Back.BLUE+Fore.WHITE+"Device name:", Back.YELLOW,Fore.BLACK, device.name)
	print(Back.BLUE+Fore.WHITE+"Device type: ", Back.YELLOW,Fore.BLACK,cl.device_type.to_string(device.type),Style.RESET_ALL)
	print(Back.BLUE+"Device memory: ",Back.YELLOW,Fore.BLACK, format(device.global_mem_size//1024//1024,',d'), 'MB',Style.RESET_ALL)
	print(Back.BLUE+"Device max clock speed: ",Back.YELLOW,Fore.BLACK, format(device.max_clock_frequency,',d'), 'MHz',Style.RESET_ALL)
	print(Back.BLUE+"Device compute units: ",Back.YELLOW,Fore.BLACK, device.max_compute_units,Style.RESET_ALL)
	print(Back.BLUE+"Device max work group size: ",Back.YELLOW,Fore.BLACK, device.max_work_group_size,Style.RESET_ALL)
	print(Back.BLUE+"Device max work item sizes: ",Back.YELLOW,Fore.BLACK, device.max_work_item_sizes,Style.RESET_ALL)
	print(Fore.BLACK+Back.WHITE+"----------------------------------------------------------"+Style.RESET_ALL)

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
	in_bytes = num_vals*32.0/8.0

	# Size of data in kBytes
	in_Kbytes = in_bytes/1000

	# Size of data in MBytes
	in_Mbytes = in_Kbytes/1000

	in_Gbytes = in_Mbytes/1000

	# Check to see what the most relivent size is
	if (in_bytes < 1000):
		return str(round(in_bytes,4)) + ' B'
	elif(in_Kbytes < 1000):
		return str(round(in_Kbytes,4)) + ' kB'
	elif(in_Mbytes < 1000):
		return str(round(in_Mbytes,4)) + ' MB'
	else:
		return str(round(in_Gbytes,4)) + ' GB'

# Verify the calculated data Very slow for large data sets
def verify_feed_forward():
	print(Back.RED + Fore.YELLOW + ' Debug mode enabled! '+Back.BLUE+ Fore.WHITE + ' Verifying Feed Forward Calculation... \n' + Fore.YELLOW + Back.RED +' WARNING! ' +Fore.BLACK +Back.YELLOW+' This is very slow for large datasets! ' + Style.RESET_ALL)
	Debug_output = network_hidden
	Debug_output_1 = np.zeros(network_hidden.shape[0]).astype(np.float32)
	Debug_output_2 = network_output_weights
	Debug_output_3 = np.zeros(network_output_weights.shape[0]).astype(np.float32)

	print(Back.BLUE+'Verification opperation: ' + Back.YELLOW + Fore.BLACK +' (1/4) ' + Style.RESET_ALL)
	for i in range(0,len(network_input)):
		Debug_output[:,i] = Debug_output[:,i] * network_input[i]
	print(Back.BLUE+'Verification opperation: ' + Back.YELLOW +Fore.BLACK +' (2/4) '+ Style.RESET_ALL)	
	for i in range(0,network_hidden.shape[0]):
		Debug_output_1[i] = sum(Debug_output[i,:])
	print(Back.BLUE+'Verification opperation: ' + Back.YELLOW +Fore.BLACK +' (3/4) '+ Style.RESET_ALL)
	for i in range(0,len(Debug_output_1)):
		Debug_output_2[:,i] = Debug_output_2[:,i] * Debug_output_1[i]
	print(Back.BLUE+'Verification opperation: ' + Back.YELLOW +Fore.BLACK +' (4/4) '+ Style.RESET_ALL)
	for i in range(0,network_output_weights.shape[0]):
		Debug_output_3[i] = sum(Debug_output_2[i,:])

	#print(Debug_output_3)
	sum_current = sum(Debug_output_3)

	
	if (sum(Output) - sum_current == 0):
		print(Back.GREEN+' Verification Sucessfull! ' + Style.RESET_ALL)
	else:
		print(Back.RED+Fore.YELLOW+' ERROR! '+Back.YELLOW + Fore.BLACK+' Computation unsucessfull! ' + Style.RESET_ALL +  '\n' + Back.BLUE + 'Difference: ' +Back.YELLOW +Fore.BLACK+ str(sum(Output)-sum_current) + ' ' + Style.RESET_ALL)

# forward propigate the input data through the network
def feed_forward(input_vec,input_matrix,time):

	global network_output_weights

	# Create a command queue
	queue = cl.CommandQueue(context)
	#print('Input Vector: with dim '  + str(input_vec.shape) +'\n' + str(input_vec))
	#print('Weights: with dim' + str(input_matrix.shape) + '\n'+ str(input_matrix))

	# Find max local work group size
	max_work_group_size = min(cl_device_work_group_max_size)

	# calculte the number of work groups per row
	num_work_groups_per_row = int(input_matrix.shape[1]/max_work_group_size)+1

	if (int(input_matrix.shape[1]/max_work_group_size) == 0):
		num_work_groups_per_row = 1
	

	# Initalize a variable for padding
	remainder_padding = 0

	# Make a temp variable we can add the padding to and not affect the network variable
	padded_matrix_tmp = input_matrix

	# If there was no padding added and there is only one work group per row
	if (num_work_groups_per_row != 1):

		# Calculate how much padding is necessary to fill the last work group
		remainder_padding = abs(max_work_group_size*(num_work_groups_per_row) - input_matrix.shape[1])

		# Generate a matrix with same number of rows as hidden network and number of collums defined by remainder_padding
		insert_padding = np.zeros((input_matrix.shape[0],remainder_padding)).astype(np.float32)

		# Make a temporary version of input_matrix and add padding (zeros) so the last workgroup is filled
		padded_matrix_tmp = np.append(padded_matrix_tmp,insert_padding,axis=1)
		
		# Check if the zeros were added to the last work group correctly
		average_work_group_size_test = padded_matrix_tmp.shape[1]/(1.0*num_work_groups_per_row)

		if (int(average_work_group_size_test) != 256):
			print('ERROR: Padding might now have been allocated correctly to fill up the last work group')
			exit(1)

	# create a buffer to store the sub sums for each row
	sum_bridge = np.zeros((input_matrix.shape[0],num_work_groups_per_row)).astype(np.float32)

	# move the buffer to the device
	sum_bridge_to_device = cl_array.to_device(queue,sum_bridge.flatten())

	# Move the number of sums per row
	sums_per_row_to_device = cl_array.to_device(queue,sum_bridge.shape[1]*np.ones(1).astype(np.int))

	#local_work_size = (input_matrix.shape[1]/num_work_groups_per_row,1)
	local_work_size = (0,0)
	if (padded_matrix_tmp.shape[1] <= 256):
		local_work_size = (padded_matrix_tmp.shape[1],1)
	else:
		local_work_size = (256,1)

	# Move data to device and create a pointer to it.
	hidden_width_to_device = cl_array.to_device(queue,padded_matrix_tmp.shape[1]*np.ones(1).astype(np.uint))
	network_hidden_to_device = cl_array.to_device(queue, padded_matrix_tmp.flatten())
	network_input_to_device = cl_array.to_device(queue, input_vec)
	opperation_output_to_device = cl_array.empty(queue, input_matrix.shape[0], dtype=np.float32)
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
								opperation_output_to_device.data,
								sum_bridge_to_device.data,
								sum_local_to_device)


	sum_bridge_get = sum_bridge_to_device.get()
	#print('Sum Bridge results: with dims' + str(sum_bridge_get.shape) + '\n' + str(sum_bridge_get))
	# Get the output from the device
	if (time == 1):
		return opperation_output_to_device.get()
	else:
		return feed_forward(opperation_output_to_device.get(),network_output_weights,1)


# initalize a timer object to time the calculation
t = Timer()

print(Back.BLUE+'Generating input data...'+ Style.RESET_ALL)
load_input_data('random')

print(Back.BLUE+'Creating the data structure...'+ Style.RESET_ALL)
init_data_structure()

print(Back.BLUE+'Finding OpenCL Devices...'+ Style.RESET_ALL)
cl_find_devices()

print(Back.BLUE+'Generating OpenCL context...'+ Style.RESET_ALL)
context = cl_get_context()

print_network_information()

t.start()

print(Back.BLUE+'Feeding forward network...'+ Style.RESET_ALL)
Output = feed_forward(network_input,network_hidden,0)
#print('feed forward function output: \n'+str(Output))
#print('Output vector: with dim: ' + str(Output.shape)+ '\n'+ str(Output))


t.print_elapsed_time()


# If debugging mode is inabled verify the opencl calculation
if (DEBUG):
	t.reset()
	verify_feed_forward()
	t.print_elapsed_time()















