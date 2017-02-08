from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import os
import sys
import colorama
from colorama import Fore, Back, Style
import platform
from psutil import virtual_memory
import time


# Clear the screen
if (platform.system() == 'Windows'):
	os.system('cls')
else:
	os.system('clear')

# Timer Class for having instances of a timer for benchmarking
class Timer(object):

	

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
		seconds = time.time() - self.t
		minutes = 0
		while (seconds > 60):
				minutes = minutes + 1
				seconds = seconds - 60
		if (minutes == 0):
			time_elapsed = str(round(seconds,5)) + ' seconds'
		else:
			time_elapsed = str(minutes) + ' minutes ' + str(round(seconds,5)) + ' seconds '

		print(Back.BLUE+ msg +Back.YELLOW +Fore.BLACK+ ' '+ time_elapsed + ' '+ Style.RESET_ALL)

	def reset(self):
		self.t = time.time()		

# Neural Network Class
class Neural_Net(object):
	"""docstring for Neural_Net"""

	def __init__(self,net_input_size,net_hidden_size,net_output_size):

		# Initilize colorama to get console colors
		colorama.init()

		# Debug settings
		self.DEBUG_feed_forward = False
		self.DEBUG_feed_forward_verification = False
		self.DEBUG_cl_devices = False
		self.DEBUG_network_info = False

		# OpenCL properties
		self.target_cl_device_type = cl.device_type.GPU
		self.cl_device_list = []
		self.cl_device_work_group_max_size = []
		self.context = 0

		# Network size properties
		self.input_size = net_input_size
		self.hidden_size = net_hidden_size
		self.output_size = net_output_size

		# Network storage variables
		self.network_input = []
		self.network_hidden = []
		self.network_output = []
		self.network_output_weights = []

	def net_full_debug(self):
		self.DEBUG_feed_forward = True
		self.DEBUG_feed_forward_verification = True
		self.DEBUG_cl_devices = True
		self.DEBUG_network_info = True

	# Create the hidden and output data structure with numpy arrays
	def init_data_structure(self):
		print(Back.BLUE+'Creating the data structure...'+ Style.RESET_ALL)
		self.network_hidden = np.ones((self.hidden_size,self.input_size)).astype(np.float32)
		self.network_output = (np.zeros(self.output_size)).astype(np.float32)
		self.network_output_weights = np.ones((self.output_size,self.hidden_size)).astype(np.float32)

	# Load some input data to feed to the network
	def load_input_data(self,data_type):
		print(Back.BLUE+'Generating input data...'+ Style.RESET_ALL)
		if data_type == 'random':
			self.network_input = 1*np.ones(self.input_size).astype(np.float32)
		elif data_type == 'from':
			return 0

	def print_network_information(self):
		if (self.DEBUG_network_info):
			print(Fore.BLACK + Back.WHITE+'==========================================================')
			print('=======' + Back.GREEN + Fore.BLACK+'            Network Information            '+Back.WHITE+Fore.BLACK+'========')
			print('==========================================================' + Style.RESET_ALL)
			print(Fore.WHITE+Back.BLUE+'Input size: ' + Fore.BLACK+Back.YELLOW+ ' '+ str(format(self.input_size,',d'))+ ' ' +Style.RESET_ALL)
			print(Fore.WHITE+Back.BLUE+'Hidden size: ' + Fore.BLACK+Back.YELLOW+' '+str(format(self.hidden_size,',d'))+ ' '+ Fore.BLACK+Back.YELLOW+Style.RESET_ALL)
			print(Fore.WHITE+Back.BLUE+'Output size: ' + Fore.BLACK+Back.YELLOW+' '+str(format(self.output_size,',d'))+  ' ' +Style.RESET_ALL)
			print(Fore.WHITE+Back.BLUE+'Number of data points: '  +Fore.BLACK+Back.YELLOW+ ' '+str(format(self.input_size + self.input_size*self.hidden_size + self.output_size,',d'))+ ' '+Fore.BLACK+Back.YELLOW+ Style.RESET_ALL)
			print(Fore.WHITE+Back.BLUE+'Estimated on device data size: ' +Fore.BLACK+Back.YELLOW+ ' '+self.estimate_vram_usage() + ' '+Fore.BLACK+Back.YELLOW+ Style.RESET_ALL)
			print(Fore.BLACK + Back.WHITE+'==========================================================')
			print('======='+ Back.GREEN + Fore.BLACK+'            Current Computation            '+Back.WHITE+Fore.BLACK+'========')
			print('=========================================================='+ Style.RESET_ALL)

	# Find and cataloge all of the opencl compatable devices on the system
	def cl_find_devices(self):
		print(Back.BLUE+'Finding OpenCL Devices...'+ Style.RESET_ALL)
		plats = cl.get_platforms()
		for plat in plats:
			index = plats.index(plat)
			devices = plats[index].get_devices(self.target_cl_device_type)
			for device in devices: 
				self.cl_device_list.append(device)
				self.cl_device_work_group_max_size.append(device.max_work_group_size)
		if (self.DEBUG_cl_devices):
			print(Fore.BLACK + Back.WHITE+'==========================================================')
			print('======='+ Back.GREEN + Fore.BLACK+'      OpenCL Devices on this platform      '+Back.WHITE+Fore.BLACK+'========')
			print('=========================================================='+ Style.RESET_ALL)
			print(Back.BLUE + Fore.WHITE+'Number of OpenCl devices found: '+ Back.YELLOW + Fore.BLACK +' '+ str(len(self.cl_device_list)) + ' ' + Style.RESET_ALL)
			for device in self.cl_device_list:
				num = self.cl_device_list.index(device)
				self.cl_print_device_information(device,num)

	# Get the context for a given device
	def cl_get_context(self):
		print(Back.BLUE+'Generating OpenCL context...'+ Style.RESET_ALL)
		context_found = cl.Context(self.cl_device_list)
		self.context = context_found

	# Load an opencl kenrel file as a string
	def cl_load_kernel(self,name):
		kernel = open(name,'r').read() 
		return kernel

	# Print out information about a specific opencl device
	def cl_print_device_information(self,device,number):
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
	def estimate_vram_usage(self):

		# Number of values for hidden matrix
		num_vals_comp_1 = self.network_hidden.shape[0] * self.network_hidden.shape[1]
		
		# Number of values for input vector
		num_vals_comp_1 = num_vals_comp_1 + max(self.network_input.shape)
		
		# Number of values for hidden vector intermediate output
		num_vals_comp_1 = num_vals_comp_1 + self.network_hidden.shape[0]
		
		# Number of values for the local buffer 
		num_vals_comp_1 = num_vals_comp_1 + max(self.network_hidden[0,:].shape)

		num_vals_comp_2 = self.network_output_weights.shape[0]*self.network_output_weights.shape[1]

		num_vals_comp_2 = num_vals_comp_2 + self.network_output_weights.shape[0]

		num_vals_comp_2 = num_vals_comp_2 + self.network_output_weights.shape[1]

		num_vals_comp_2 = num_vals_comp_2 + max(self.network_output_weights[0,:].shape)

		num_vals = max(num_vals_comp_1,num_vals_comp_2)

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
	def verify_feed_forward(self):
		if (self.DEBUG_feed_forward_verification):
			t_verify = Timer()
			t_verify.start()

			# Notify user Debug mode is enabled and that the verification is occuring.
			print(Back.RED + Fore.YELLOW + ' Debug mode enabled! '+Back.BLUE+ Fore.WHITE + ' Verifying Feed Forward Calculation... \n' + Fore.YELLOW + Back.RED +' Warning! ' +Fore.BLACK +Back.YELLOW+' This is very slow for large datasets! ' + Style.RESET_ALL)
			
			# Define matricies to do the verification computations
			Debug_output = self.network_hidden
			Debug_output_1 = np.zeros(self.network_hidden.shape[0]).astype(np.float32)
			Debug_output_2 = self.network_output_weights
			Debug_output_3 = np.zeros(self.network_output_weights.shape[0]).astype(np.float32)

			# Multiply the input vector by the collums of the weight matrix
			print(Back.BLUE+'Verification: ' + Back.YELLOW + Fore.BLACK +' (1/4) ' + Style.RESET_ALL)
			for i in range(0,len(self.network_input)):
				Debug_output[:,i] = Debug_output[:,i] * self.network_input[i]

			# Sum each row of the result of the previous computation to get the hidden results
			print(Back.BLUE+'Verification: ' + Back.YELLOW +Fore.BLACK +' (2/4) '+ Style.RESET_ALL)	
			for i in range(0,self.network_hidden.shape[0]):
				Debug_output_1[i] = sum(Debug_output[i,:])

			# Apply the activation function to the results

			# Take the outputs of the hidden neurons and multiply the weights by thoes results
			print(Back.BLUE+'Verification: ' + Back.YELLOW +Fore.BLACK +' (3/4) '+ Style.RESET_ALL)
			for i in range(0,len(Debug_output_1)):
				Debug_output_2[:,i] = Debug_output_2[:,i] * Debug_output_1[i]

			# Sum each row of the result of the previous computation to get the output results
			print(Back.BLUE+'Verification: ' + Back.YELLOW +Fore.BLACK +' (4/4) '+ Style.RESET_ALL)
			for i in range(0,self.network_output_weights.shape[0]):
				Debug_output_3[i] = sum(Debug_output_2[i,:])

			# Define variables for use in error display
			index = 0
			error_found = 0
			num_allowed_errors_disp = 5

			# Go through each element of the test data and check it with each there is a difference
			for element in self.network_output.tolist():

				# If the values dont mach
				if element != Debug_output_3[index]:

					# If this is the first error that has been found notify
					if (error_found == 0):
						print(Back.RED+Fore.YELLOW+' Error! '+Back.YELLOW + Fore.BLACK+' Computation unsucessfull! ' + Style.RESET_ALL)
						print(Fore.BLACK + Back.WHITE+'===================================')
						print('======='+ Back.RED + Fore.YELLOW+'    Error Report    '+Back.WHITE+Fore.BLACK+'========')
						print('==================================='+ Style.RESET_ALL)
						error_found = 1

					# If the number of errors thats been displayed is below the max allowed
					if (error_found <= num_allowed_errors_disp):

						# Give info on error
						print(Back.RED,Fore.YELLOW,'          Value Error!         ',Style.RESET_ALL)
						error_padding = 12
						print(Back.BLUE,Fore.WHITE,'      index:   ',Back.YELLOW,Fore.BLACK,format(int(index),',d'),' '*(error_padding-len(str(format(int(index),',d')))),Style.RESET_ALL)
						print(Back.BLUE,Fore.WHITE,'   With value: ',Back.YELLOW,Fore.BLACK, format(int(element),',d'),' '*(error_padding-len(str(format(int(element),',d')))),Style.RESET_ALL)
						print(Back.BLUE,Fore.WHITE,'Expected value:',Back.YELLOW,Fore.BLACK,format(int(Debug_output_3[index]),',d'),' '*(error_padding-len(str(format(int(Debug_output_3[index]),',d')))),Style.RESET_ALL)
						print(Back.BLUE,Fore.WHITE,'   Difference: ',Back.YELLOW,Fore.BLACK,format(int(abs(Debug_output_3[index]-element)),',d'),' '*(error_padding-len(str(format(int(abs(Debug_output_3[index]-element)),',d')))),Style.RESET_ALL)
						if (DEBUG_feed_forward_verification):
							t_verify.print_elapsed_time_msg('Verification Time:')
					# increment the number of errors found
					error_found = error_found + 1
				# Move to the next data index
				index = index + 1

			# If the max number of errors have been seen notify 
			if (error_found > num_allowed_errors_disp):
				print(Back.RED,Fore.YELLOW, 'Error!',Back.YELLOW,Fore.BLACK, 'Number of Value Errors exceded the display limit!',Style.RESET_ALL)
				
			# Sum the vector of both the verifycation and output
			sum_current = sum(Debug_output_3)
			sum_output = sum(self.network_output)
			
			# If the sums are the same notify that the calculation was sucessful
			if (sum_output - sum_current == 0):
				print(Back.GREEN+' Verification Sucessfull! ' + Style.RESET_ALL)
				t_verify.print_elapsed_time_msg('Verification Time:')
			else:
				# Give other information about the errors
				print(Back.RED,Fore.YELLOW,'           Error Impact                     ',Style.RESET_ALL)
				error_padding = 15
				error_found = error_found-1
				print(Back.BLUE,Fore.WHITE, '     Number of errors:   ', Back.YELLOW,Fore.BLACK,format(error_found,',d'),' '*(error_padding-len(str(format(error_found,',d')))),Style.RESET_ALL)
				print(Back.BLUE,Fore.WHITE, ' Sum of GPU Calculation: ' ,Back.YELLOW,Fore.BLACK,format(int(sum_output),',d'),' '*(error_padding-len(str(format(int(sum_output),',d')))),Style.RESET_ALL)
				print(Back.BLUE,Fore.WHITE, 'Verifycation Calculation:' ,Back.YELLOW,Fore.BLACK,format(int(sum_current),',d'),' '*(error_padding-len(str(format(int(sum_current),',d')))),Style.RESET_ALL)
				print(Back.BLUE,Fore.WHITE, '       Difference:       ' ,Back.YELLOW,Fore.BLACK,format(int(abs(sum_output-sum_current)),',d'),' '*(error_padding-len(str(format(int(abs(sum_output-sum_current)),',d')))),Style.RESET_ALL)
				t_verify.print_elapsed_time_msg('Verification Time:')

	# forward propigate the input data through the network
	def feed_forward(self,input_vec,input_matrix,time):

		if (self.DEBUG_feed_forward):
			t_feed = Timer()
			t_feed.start()

		if(time == 0):
			print(Back.BLUE+'Feeding forward network...'+ Style.RESET_ALL)

		# Create a command queue
		queue = cl.CommandQueue(self.context)

		# Find max local work group size
		max_work_group_size = min(self.cl_device_work_group_max_size)

		# calculte the number of work groups per row
		num_work_groups_per_row = int(input_matrix.shape[1]/max_work_group_size)+1

		if (int(input_matrix.shape[1]/max_work_group_size) == 0):
			num_work_groups_per_row = 1
		
		if(float(input_matrix.shape[1])/(max_work_group_size)*1.0 == 1.0):
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
		program = cl.Program(self.context,self.cl_load_kernel('feed_forward.c')).build()

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
		sum_bridge_get = np.resize(sum_bridge_get,sum_bridge.shape)

		if (num_work_groups_per_row != 1):
			output_sums = self.cl_sum_rows(sum_bridge_get)
		else:
			output_sums = opperation_output_to_device.get()

		# Get the output from the device
		if (time == 1):
			return output_sums
		else:
			self.network_output = self.feed_forward(output_sums,self.network_output_weights,1)
			print(Back.BLUE+Fore.WHITE+'Feed forward Complete!',Style.RESET_ALL)
			if (self.DEBUG_feed_forward):
				t_feed.print_elapsed_time_msg('Feed forward time:')

	# Calculate the sum of the workgroups for one collum
	def cl_sum_rows(self,input_matrix):
		queue = cl.CommandQueue(self.context)

		input_matrix_to_device = cl_array.to_device(queue,input_matrix.flatten())
		input_matrix_width_to_device = cl_array.to_device(queue,input_matrix.shape[1]*np.ones(1).astype(np.uint))
		output_vector_to_device = cl_array.empty(queue,input_matrix.shape[0],dtype=np.float32)

		global_work_size = (input_matrix.shape[1],input_matrix.shape[0])

		if (input_matrix.shape[1] > 256):
			size_1 = int(input_matrix.shape[1]/2)
			size_2 = input_matrix.shape[1]-size_1

			part_1 = cl_sum_rows(input_matrix[:,0:size_1])
			part_2 = cl_sum_rows(input_matrix[:,size_1:input_matrix.shape[1]])

			togther=np.append(part_1[None].T,part_2[None].T,axis=1)

			return cl_sum_rows(togther)

		local_work_size = (input_matrix.shape[1],1)

		program = cl.Program(self.context,self.cl_load_kernel('sum_rows.c')).build()

		program.sum_rows(queue,global_work_size,local_work_size,input_matrix_to_device.data,input_matrix_width_to_device.data,output_vector_to_device.data)

		return output_vector_to_device.get()


# Initalize Network with Neural_Net(input_size,hidden_size,output_size)
n = Neural_Net(1000,1000,300)

n.net_full_debug()

n.load_input_data('random')

n.init_data_structure()

n.cl_find_devices()

n.cl_get_context()

n.print_network_information()


n.feed_forward(n.network_input,n.network_hidden,0)


n.verify_feed_forward()
















