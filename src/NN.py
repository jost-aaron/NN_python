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
import math
from random import randint


os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

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
		self.DEBUG_forward_prop = False
		self.DEBUG_forward_prop_verification = False
		self.DEBUG_forward_prop_verification_in_progress = False
		self.DEBUG_cl_devices = False
		self.DEBUG_network_info = False

		# OpenCL properties
		self.target_cl_device_type = cl.device_type.GPU
		self.cl_device_list = []
		self.cl_device_work_group_max_size = []
		self.context = 0
		self.cl_num_opps = 0

		# Network size properties
		self.input_size = net_input_size
		self.hidden_size = net_hidden_size
		self.output_size = net_output_size

		# Network learning rate
		self.learning_rate = 3

		# Network Partial Derivatives
		self.dJdW1 = 0
		self.dJdW2 = 0

		# Network storage variables
		self.network_input = []
		self.network_hidden = []
		self.network_hidden_activation = []
		self.network_hidden_activity = []
		self.network_output = []
		self.network_output_weights = []
		self.network_output_activation = []

		# Network activation function
		self.network_activation_function = 0
		self.network_activation_function_num = 0
		

	def net_full_debug(self):
		self.DEBUG_forward_prop = True
		self.DEBUG_forward_prop_verification = True
		self.DEBUG_cl_devices = True
		self.DEBUG_network_info = True

	# Create the hidden and output data structure with numpy arrays
	def init_data_structure(self):
		print(Back.BLUE+'Creating the data structure...'+ Style.RESET_ALL)
		self.network_hidden = np.random.rand(self.hidden_size,self.input_size).astype(np.float32)
		self.network_output = (np.zeros(self.output_size)).astype(np.float32)
		self.network_output_weights = np.random.rand(self.output_size,self.hidden_size).astype(np.float32)

	# Load some input data to feed to the network
	def load_input_data(self,data_type):
		print(Back.BLUE+'Generating input data...'+ Style.RESET_ALL)
		if data_type == 'random':
			self.network_input = 1*np.random.rand(self.input_size).astype(np.float32)
		elif data_type == 'from':
			return 0

	# Print out information about the networks size
	def print_network_information(self):
		vram_estimate = self.estimate_vram_usage()
		if (self.DEBUG_network_info):
			print(Fore.BLACK + Back.WHITE+'==========================================================')
			print(Fore.BLACK + Back.WHITE+'=======' + Back.GREEN + Fore.BLACK+'            Network Information            '+Back.WHITE+Fore.BLACK+'========')
			print('==========================================================' + Style.RESET_ALL)
			print(Fore.WHITE+Back.BLUE+'Input size: ' + Fore.BLACK+Back.YELLOW+ ' '+ str(format(self.input_size,',d'))+ ' ' +Style.RESET_ALL)
			print(Fore.WHITE+Back.BLUE+'Hidden size: ' + Fore.BLACK+Back.YELLOW+' '+str(format(self.hidden_size,',d'))+ ' '+ Fore.BLACK+Back.YELLOW+Style.RESET_ALL)
			print(Fore.WHITE+Back.BLUE+'Output size: ' + Fore.BLACK+Back.YELLOW+' '+str(format(self.output_size,',d'))+  ' ' +Style.RESET_ALL)
			print(Fore.WHITE+Back.BLUE+'Number of data points: '  +Fore.BLACK+Back.YELLOW+ ' '+str(format(self.input_size + self.input_size*self.hidden_size + self.output_size + self.output_size*self.hidden_size,',d'))+ ' '+Fore.BLACK+Back.YELLOW+ Style.RESET_ALL)
			print(Fore.WHITE+Back.BLUE+'Estimated on device data size: ' +Fore.BLACK+Back.YELLOW+ ' '+vram_estimate[0] + ' '+Fore.BLACK+Back.YELLOW+ Style.RESET_ALL)
		try:
			if (vram_estimate[1][0] == 1):
				errors = vram_estimate[1][1:]
				for error in errors:
					padding = 32
					print(Fore.YELLOW+Back.RED,'Warning!',Fore.BLACK+Back.YELLOW,'The following devices have insuficent VRAM for the current computation:',Style.RESET_ALL)
					print(Fore.YELLOW+Back.RED,errors.index(error)+1,Fore.WHITE,Back.BLUE,'   Device:',Fore.BLACK+Back.YELLOW,error[0],' '*(padding-len(str(error[0]))),Style.RESET_ALL)
					print(Fore.YELLOW+Back.RED,' ',Fore.WHITE,Back.BLUE,'Avaliable:',Fore.BLACK+Back.YELLOW,str(error[1])+' MB',' '*(padding-len(str(error[1])+' MB')),Style.RESET_ALL)
					print(Fore.YELLOW+Back.RED,' ',Fore.WHITE,Back.BLUE,' Required:',Fore.BLACK+Back.YELLOW,str(error[2])+' MB',' '*(padding-len(str(error[2])+' MB')),Style.RESET_ALL)
		except:
			pass
		if (self.DEBUG_network_info):
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

	# Calculate the sum of the workgroups for one collum
	def cl_sum_rows(self,input_matrix,eval_activation):
		queue = cl.CommandQueue(self.context)

		if (self.DEBUG_forward_prop):
			self.cl_num_opps = self.cl_num_opps+1

		global_work_size = (input_matrix.shape[1],input_matrix.shape[0])

		if (input_matrix.shape[1] > 256):
			size_1 = int(input_matrix.shape[1]/2)
			size_2 = input_matrix.shape[1]-size_1

			part_1 = cl_sum_rows(input_matrix[:,0:size_1],0)
			part_2 = cl_sum_rows(input_matrix[:,size_1:input_matrix.shape[1]],0)

			togther=np.append(part_1[None].T,part_2[None].T,axis=1)

			return cl_sum_rows(togther)

		local_work_size = (input_matrix.shape[1],1)

		input_matrix_to_device = cl_array.to_device(queue,input_matrix.flatten())
		input_matrix_width_to_device = cl_array.to_device(queue,input_matrix.shape[1]*np.ones(1).astype(np.uint))
		output_vector_to_device = cl_array.empty(queue,input_matrix.shape[0],dtype=np.float32)
		eval_activation_to_device = cl_array.to_device(queue,eval_activation*np.ones(1).astype(np.uint))
		activation_function_to_device = cl_array.to_device(queue,self.network_activation_function_num*np.ones(1).astype(np.uint))


		program = cl.Program(self.context,self.cl_load_kernel('sum_rows.c')).build()

		program.sum_rows(queue,global_work_size,local_work_size,input_matrix_to_device.data,input_matrix_width_to_device.data,output_vector_to_device.data,eval_activation_to_device.data,activation_function_to_device.data)

		return output_vector_to_device.get()

	# Initialize OpenCL stuff
	def cl_init(self):
		self.cl_find_devices()
		self.cl_get_context()

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

		flag = 0

		for device in self.cl_device_list:
			device_global_mem = device.global_mem_size//1024//1024
			if (in_Mbytes > device_global_mem):
			#if (1600 > device_global_mem):
				if (flag == 0):
					flag = list((1,list((device.name,int(device_global_mem),round(in_Mbytes,4)))))
				else:
					flag.append(list((device.name,int(device_global_mem),round(in_Mbytes,4))))

		# Check to see what the most relivent size is
		if (in_bytes < 1000):
			return list((str(round(in_bytes,4)) + ' B',flag))
		elif(in_Kbytes < 1000):
			return list((str(round(in_Kbytes,4)) + ' kB',flag))
		elif(in_Mbytes < 1000):
			return list((str(round(in_Mbytes,4)) + ' MB',flag))
		else:
			return list((str(round(in_Gbytes,4)) + ' GB',flag))

	# Verify the calculated data Very slow for large data sets
	def verify_forward_prop(self):
		if (self.DEBUG_forward_prop_verification):
			t_verify = Timer()
			t_verify.start()

			Debug_output_3 = forward_prop_cpu();

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

						error_padding = 16
						# Give info on error
						print(Back.RED,Fore.YELLOW,'          Value Error!',' '*(error_padding-4),Style.RESET_ALL)
						print(Back.BLUE,Fore.WHITE,'      index:   ',Back.YELLOW,Fore.BLACK,format(int(index),',d'),' '*(error_padding-len(str(format(int(index),',d')))),Style.RESET_ALL)
						print(Back.BLUE,Fore.WHITE,'   With value: ',Back.YELLOW,Fore.BLACK, format(element,',f'),' '*(error_padding-len(str(format(element,',f')))),Style.RESET_ALL)
						print(Back.BLUE,Fore.WHITE,'Expected value:',Back.YELLOW,Fore.BLACK,format(Debug_output_3[index],',f'),' '*(error_padding-len(str(format(Debug_output_3[index],',f')))),Style.RESET_ALL)
						print(Back.BLUE,Fore.WHITE,'   Difference: ',Back.YELLOW,Fore.BLACK,format(abs(Debug_output_3[index]-element),',f'),' '*(error_padding-len(str(format(abs(Debug_output_3[index]-element),',f')))),Style.RESET_ALL)
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
			else:
				# Give other information about the errors
				error_padding = 20
				print(Back.RED,Fore.YELLOW,'           Error Impact',' '*(error_padding+5),Style.RESET_ALL)
				error_found = error_found-1
				print(Back.BLUE,Fore.WHITE, '     Number of errors:   ', Back.YELLOW,Fore.BLACK,format(int(error_found),',d'),' '*(error_padding-len(str(format(int(error_found),',d')))),Style.RESET_ALL)
				print(Back.BLUE,Fore.WHITE, ' Sum of GPU Calculation: ' ,Back.YELLOW,Fore.BLACK,format(sum_output,',f'),' '*(error_padding-len(str(format(sum_output,',f')))),Style.RESET_ALL)
				print(Back.BLUE,Fore.WHITE, 'Verifycation Calculation:' ,Back.YELLOW,Fore.BLACK,format(sum_current,',f'),' '*(error_padding-len(str(format(sum_current,',f')))),Style.RESET_ALL)
				print(Back.BLUE,Fore.WHITE, '       Difference:       ' ,Back.YELLOW,Fore.BLACK,format(abs(sum_output-sum_current),',f'),' '*(error_padding-len(str(format(abs(sum_output-sum_current),',f')))),Style.RESET_ALL)
			if (self.DEBUG_forward_prop_verification):
				t_verify.print_elapsed_time_msg('Verification Time:')

	# forward propigate the input data through the network.
	#n.forward_prop_cl(n.network_input,n.network_hidden,0)
	def forward_prop_cl(self,input_vec,input_matrix,time):
		t_feed = Timer()
		t_feed.start()

		if (self.DEBUG_forward_prop):
			self.cl_num_opps = self.cl_num_opps+1

		if(time == 0):
			# Check if the activation function has been specifyed
			if (self.network_activation_function == 0):
				print(Fore.YELLOW+Back.RED,'Warning!',Fore.BLACK+Back.YELLOW,'Activation function not specifyed! Using Default: HTan.'+Style.RESET_ALL)
			# If it has been set change the number to the correct number
			else:
				if (self.network_activation_function == 'HTan'):
					self.network_activation_function_num = 0
				elif (self.network_activation_function == 'Logistic'):
					self.network_activation_function_num = 1

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
		program = cl.Program(self.context,self.cl_load_kernel('forward_prop.c')).build()

		# Call the kernel and load arguments
		program.forward_prop(queue,
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
			output_sums = self.cl_sum_rows(sum_bridge_get,1)
		else:
			output_sums = opperation_output_to_device.get()

		# Get the output from the device
		if (time == 1):
			return output_sums
		else:
			self.network_output = self.forward_prop_cl(output_sums,self.network_output_weights,1)
			print(Back.BLUE+Fore.WHITE+'Feed forward Complete!',Style.RESET_ALL)
			if (self.DEBUG_forward_prop):
				t_feed.print_elapsed_time_msg('Feed forward time:')

	def forward_prop_cpu(self):
		# Define matricies to do the verification computations
			Debug_output = self.network_hidden
			Debug_output_1 = np.zeros(self.network_hidden.shape[0]).astype(np.float32)
			Debug_output_2 = self.network_output_weights
			Debug_output_3 = np.zeros(self.network_output_weights.shape[0]).astype(np.float32)


			if (self.DEBUG_forward_prop_verification_in_progress):
				print(Back.BLUE+'Verification: ' + Back.YELLOW + Fore.BLACK +' (1/6) ' + Style.RESET_ALL)
			
			# Multiply the input vector by the collums of the weight matrix
			for i in range(0,len(self.network_input)):
				Debug_output[:,i] = Debug_output[:,i] * self.network_input[i]

			if (self.DEBUG_forward_prop_verification_in_progress):
				print(Back.BLUE+'Verification: ' + Back.YELLOW +Fore.BLACK +' (2/6) '+ Style.RESET_ALL)	
			
			# Sum each row of the result of the previous computation to get the hidden results
			for i in range(0,self.network_hidden.shape[0]):
				Debug_output_1[i] = sum(Debug_output[i,:])

			# Save hidden layer activation
			self.network_hidden_activation = Debug_output_1

			if (self.DEBUG_forward_prop_verification_in_progress):
				print(Back.BLUE+'Verification: ' + Back.YELLOW +Fore.BLACK +' (3/6) '+ Style.RESET_ALL)
			
			# Apply the activation function to the results	
			# Activation function HTan
			if (self.network_activation_function_num == 0):
				Debug_output_1[:] = (2/(1+np.exp(-2*Debug_output_1[:]))) - 1
			# Activation function logistic equation
			elif (self.network_activation_function_num == 1):
				Debug_output_1[:] = (1/(1+np.exp(-1*Debug_output_1[:])))
			
			# Save hidden layer activity
			self.network_hidden_activity = Debug_output_1 

			if (self.DEBUG_forward_prop_verification_in_progress):
				print(Back.BLUE+'Verification: ' + Back.YELLOW +Fore.BLACK +' (4/6) '+ Style.RESET_ALL)
			
			# Take the outputs of the hidden neurons and multiply the weights by thoes results
			for i in range(0,len(Debug_output_1)):
				Debug_output_2[:,i] = Debug_output_2[:,i] * Debug_output_1[i]

			if (self.DEBUG_forward_prop_verification_in_progress):
				print(Back.BLUE+'Verification: ' + Back.YELLOW +Fore.BLACK +' (5/6) '+ Style.RESET_ALL)
			
			# Sum each row of the result of the previous computation to get the output results
			for i in range(0,self.network_output_weights.shape[0]):
				Debug_output_3[i] = sum(Debug_output_2[i,:])

			# Save output layer activation
			self.network_output_activation = Debug_output_3

			if (self.DEBUG_forward_prop_verification_in_progress):
				print(Back.BLUE+'Verification: ' + Back.YELLOW +Fore.BLACK +' (6/6) '+ Style.RESET_ALL)
			
			# Apply the activation function to the results	
			# Activation function HTan
			if (self.network_activation_function_num == 0):
				Debug_output_3[:] = (2/(1+np.exp(-2*Debug_output_3[:]))) - 1
			# Activation function logistic equation
			elif (self.network_activation_function_num == 1):
				Debug_output_3[:] = (1/(1+np.exp(-1*Debug_output_3[:])))

			self.network_output = Debug_output_3

			if(self.DEBUG_forward_prop_verification_in_progress):
				return Debug_output_3

	# Functions for training
	#----------------------------------------------------
	def sigmoide(self,v_in):
		return 1/(1+np.exp(-v_in))

	def sigmoid_prime(self,v_in):
		return np.exp(-v_in)/((1 + np.exp(-v_in))**2)

	def cost_function(self,v_in):
		return sum((1/2)*( known_result - self.v_in)**2)
	#----------------------------------------------------

	def back_prop_cpu(self,known_result):
		output_weights_error = self.network_output * (1 - self.network_output) * (self.network_output - known_result)
		hidden_wieghts_error = self.network_hidden_activity * (1 - self.network_hidden_activity) * sum(output_weights_error * self.network_output_weights)

		return hidden_wieghts_error,output_weights_error

	def train_grad_decent_cpu(self):

		# Temportary until we have more training capabilities
		#---------------------------------------------------
		# Max number of training itterations.
		max_itter = 1
		# Known test data to use
		known_result = np.ones(len(self.network_output))
		for i in range(0,len(known_result)):
			num = randint(0,100)
			if (num > 50):
				known_result[i] = 1
			else:
				known_result[i] = 0
		#---------------------------------------------------



		# Training loop
		for j in range(0,max_itter):

			self.forward_prop_cpu()

			hidden_wieghts_error,output_weights_error = self.back_prop_cpu(known_result)

			print('Hidden weights: \n',self.network_hidden)
			print('Hidden weights error: \n',hidden_wieghts_error)
			print('Output weights: \n', self.network_output_weights)
			print('output weights error: \n',output_weights_error)


    		

			# Change the weights
			#self.network_hidden = self.network_hidden + learning_rate*self.dJdW1
			#self.network_output_weights = self.network_output_weights + learning_rate*self.dJdW2

			#print('Current cost function value: ', cost_function(network_output))

		
	


# Initalize Network with Neural_Net(input_size,hidden_size,output_size)
n = Neural_Net(2,3,1)

n.network_activation_function = 'Logistic'

n.net_full_debug()

n.load_input_data('random')

n.init_data_structure()

n.cl_init()

n.print_network_information()

n.train_grad_decent_cpu()

















