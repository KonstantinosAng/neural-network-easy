import random
from csv import reader
import math
from math import atan
from math import exp
import numpy as np
from matplotlib import pyplot as plt
from progressbar import *
import os


class NN:

	def __init__(self, DATABASE, n_inputs, n_hidden, n_outputs, init_l_rate, epoch, optimize, loss):
		self.dir_path = os.path.dirname(os.path.realpath(__file__))		
		self.database_name = os.path.join(self.dir_path, DATABASE)
		self.n_inputs = n_inputs
		self.n_hidden = n_hidden
		self.n_outputs = n_outputs
		self.init_l_rate = init_l_rate
		self.epoch = epoch
		self.optimize = optimize
		self.loss = loss
		self.database = list()  #initialize database
		self.training_database = None
		self.testing_database = None
		self.h_out = [0 for x in range(self.n_hidden)]  # hidden layer output
		self.o_out = [0 for x in range(self.n_outputs)]  # output layer output
		self.o_cost = [0 for x in range(self.n_outputs)]  # output cost
		self.row = [0 for x in range(self.n_inputs)]  # row with input data
		self.target = [0 for x in range(self.n_outputs)]  # row with output data
		self.o_delta_cost = [0 for x in range(self.n_outputs)]  # derivative of cost for output
		self.o_delta_sigmoid = [0 for x in range(self.n_outputs)]  # derivative of sigmoid fot output
		self.o_delta = [[0 for x in range(self.n_hidden)] for y in range(self.n_outputs)]  # delta for correction
		self.u_h_weights = [[0 for x in range(self.n_inputs)] for y in range(self.n_hidden)]  # updated weights for hidden layer nodes
		self.u_o_weights = [[0 for x in range(self.n_hidden)] for y in range(self.n_outputs)]  # updated weights for output layer nodes
		self.d_E_d_out_h = [0 for x in range(self.n_outputs)]  # cost of hidden layer
		self.h_delta = [[0 for x in range(self.n_inputs)] for y in range(self.n_hidden)]  # corrections for hidden layer weights
		self.final_d_E_d_out_h = 0  # cost of hidden
		self.tr_counter = 0  # counter for training errors
		self.mean_sqrd_error = [0 for x in range(self.epoch)]  # store mean squared error
		self.rt_mean_sqrd_error = [0 for x in range(self.epoch)]  # store root mean squared error
		self.arctan_error = [0 for x in range(self.epoch)]  # store arctan error
		self.tr_mean_sqrd_error = [0 for x in range(self.epoch)]  # store training mean squared error
		self.tr_rt_mean_sqrd_error = [0 for x in range(self.epoch)]  # store training root mean squared error
		self.tr_arctan_error = [0 for x in range(self.epoch)]  # store training arctan error
		self.tr_accuracy = 0  # accuracy of training phase for each iteration
		self.sum_tr_accuracy = [0 for x in range(self.epoch)]  # accuracy of training phase for each epoch
		self.tst_accuracy = 0  # accuracy of testing phase

	# Load CSV file with database
	def load_csv(self):
		with open(self.database_name, 'r') as file:
			csv_reader = reader(file)  # open csv file with data
			for row in csv_reader:
				if not row:
					continue
				self.database.append(row)  # write each row in database

	# Calculate collumn mean
	def col_mean(self, data):
		means = [0 for i in range(self.n_inputs)]  #initialize
		for i in range(self.n_inputs):
			col_values = [float(row[i]) for row in data]  # store values of each collumn
			means[i] = sum(col_values) / float(len(data))  # calculate mean value of each collumn
		return means

	# Calculate standard deviation of collumn
	def stdv_col(self, data, means):
		stdevs = [0 for i in range(self.n_inputs)]  # initialize
		for i in range(self.n_inputs):
			variance = [pow((float(row[i]))-means[i], 2) for row in data]  #calculate variation of each collumn
			stdevs[i] = sum(variance)  # Store standard deviations of each collumn
		stdevs = [math.sqrt(x/(self.n_inputs-1)) for x in stdevs]  # caclualte the standard deviation of each collumn
		return stdevs

	#Standardize the dataset
	def standardized_data(self, data, means, stdevs):
		for row in data:
			for i in range(self.n_inputs):
				# substract the mean value and divide with standard deviation to force data values to be at range [0,1]
				row[i] = ( (float(row[i])) - means[i] ) / stdevs[i]

	#Decaying learning rate
	def dec_l_rate(self, epoch, l_rate):
		dec_rate = 0.1 # rate of decay
		a = l_rate / (1 + dec_rate*epoch)  # calculate new learning rate
		return a

	# Initialize a network ( calculate weights )
	def initialize_network(self):
		inp = [[(np.random.randn()-1) for i in range(self.n_inputs+1)] for j in range(self.n_hidden)]  #initialize weights for input
		out = [[(np.random.randn()-1) for i in range(self.n_hidden+1)] for j in range(self.n_outputs)] # initialize weights for output
		return inp, out

	# Activate neurons
	def activate(self, weights, inputs):
		activation = weights[-1]  # adding bias
		for i in range(len(inputs)):
			activation += weights[i] * inputs[i]  #calculating activation
		return activation

	#ReLu function only for hidden layer nodes
	def relu(self, activate):
		return np.maximum(0, activate)  # if the output is zero returns zero or the actual value

	# Softmax activation only for output layer nodes ( needs the sum of all output values as z vector )
	def softmax(self, z):
		for i in range(self.n_outputs):
			A += np.exp(z[i])  # sum of exp of all output values
		for i in range(self.n_outputs):
			out[i] = np.exp(z[i]) / A  #Softmax activation(like standardization) dividing by sum forces value to be at range [0,1]
		return out

	# calculate outpout based on sigmoid, alternative is tanh ( -1, 0, 1 )
	def sigmoid(self, activate):
		return 1.0 / (1.0 + np.exp(-activate))

	# cost of sigmoid for back propagation
	def d_sigmoid(self, activate):
		return self.sigmoid(activate)*(1 - self.sigmoid(activate))

	# cost of output
	def cost(self, target, out):
			return ((target - out)**2)/2

	# derivative of cost
	def d_cost(self, out, target):
		return (out - target)

	# cross entropy loss function for multiclass
	def cross_entropy(self, target, out):
		return -( target*np.log(out) + (1-target)*np.log(1-out) )

	#derivative of cross entropy loss function
	def d_cross_entropy(self, out, target):
		return -( (target/out) + (1-target)/(1-out) )

	# Adam optimizer for the hidden and output weights
	def Adam(self, dw, epoch, case):
		# Initialize hyperparameters
		beta1 = 0.9
		beta2 = 0.999
		epsilon = 1e-8
		if case == 'output':
			Vdw = [[0 for x in range(self.n_hidden)] for y in range(self.n_outputs)]  #Initialize parameters
			Sdw = [[0 for x in range(self.n_hidden)] for y in range(self.n_outputs)]  #Initialize parameters
			for i in range(self.n_outputs):
				for j in range(self.n_hidden):
					Vdw[i][j] = beta1*Vdw[i][j] + (1 - beta1)*dw[i][j]
					Sdw[i][j] = beta2*Sdw[i][j] + (1 - beta2)*(dw[i][j]**2)
					Vdw[i][j] = Vdw[i][j] / (1 - (beta1**(epoch+1)))
					Sdw[i][j] = Sdw[i][j] / (1 - (beta2**(epoch+1)))
		elif case == 'hidden':
			Vdw = [[0 for x in range(self.n_inputs)] for y in range(self.n_hidden)]  # Initialize parameters
			Sdw = [[0 for x in range(self.n_inputs)] for y in range(self.n_hidden)]  # Initialize parameters
			for i in range(self.n_hidden):
				for j in range(self.n_inputs):
					Vdw[i][j] = beta1*Vdw[i][j] + (1 - beta1)*dw[i][j]
					Sdw[i][j] = beta2*Sdw[i][j] + (1 - beta2)*(dw[i][j]**2)
					Vdw[i][j] = Vdw[i][j] / (1 - (beta1**(epoch+1)))
					Sdw[i][j] = Sdw[i][j] / (1 - (beta2**(epoch+1)))
		return Vdw, Sdw, epsilon

	# Calculate the MEANS SQUARED ERROR, ROOT MEAN SQUARED ERROR and ARCTAN ERROR for the training database
	def tr_calc_err(self, o_weights, h_weights):
		a = 0  # var for activation
		out_h = [0 for x in range(self.n_hidden)]
		out_o = [0 for x in range(self.n_outputs)]
		inp = [0 for x in range(self.n_inputs)]
		target = [0 for x in range(self.n_outputs)]
		# Feeding forward after each epoch to calculate error
		for j in range(len(self.training_database)):
			for i in range(self.n_inputs):
				inp[i] = float(self.training_database[j][i])
			for i in range(self.n_outputs):
				target[i] = float(self.training_database[j][i+7])
			for i in range(self.n_hidden):
				a = self.activate(h_weights[i], inp)
				out_h[i] = self.sigmoid(a)
			for i in range(self.n_outputs):
				a = self.activate(o_weights[i], out_h)
				out_o[i] = self.sigmoid(a)
			for i in range(self.n_outputs):
				self.mean_sqrd_error[self.tr_counter] += (target[i] - out_o[i])**2  #calculate error
				self.arctan_error[self.tr_counter] += (atan(target[i] - out_o[i]))**2  #calculate error
		self.mean_sqrd_error[self.tr_counter] = (self.mean_sqrd_error[self.tr_counter] / len(self.training_database))*100  #store error
		self.arctan_error[self.tr_counter] = (self.arctan_error[self.tr_counter] / len(self.training_database))*100  #store error
		self.rt_mean_sqrd_error[self.tr_counter] = (self.mean_sqrd_error[self.tr_counter]**(1.0/2))  #store error
		return self.mean_sqrd_error, self.rt_mean_sqrd_error, self.arctan_error

	# Calculate the MEAN SQUARED ERROR, ROOT MEAN SQUARED ERROR and ARCTAN ERROR for the testing period
	def tst_calc_err(self, o_weights, h_weights, tst_data):
		a = 0  # var for activation
		out_h = [0 for x in range(self.n_hidden)]
		out_o = [0 for x in range(self.n_outputs)]
		inp = [0 for x in range(self.n_inputs)]
		target = [0 for x in range(self.n_outputs)]
		tst_mean_sqrd_error = 0  # store testing mean squared error
		tst_rt_mean_sqrd_error = 0  # store testing root mean squared error
		tst_arctan_error = 0  # store testing arctan error
		# Feeding forward after each epoch to calculate the error
		for j in range(len(tst_data)):
			for i in range(self.n_inputs):
				inp[i] = float(tst_data[j][i])
			for i in range(self.n_outputs):
				target[i] = float(tst_data[j][i+7])
			for i in range(self.n_hidden):
				a = self.activate(h_weights[i], inp)
				out_h[i] = self.sigmoid(a)
			for i in range(self.n_outputs):
				a = self.activate(o_weights[i], out_h)
				out_o[i] = self.sigmoid(a)
			for i in range(self.n_outputs):
				tst_mean_sqrd_error += (target[i] - out_o[i])**2  #calculate error
				tst_arctan_error += (atan(target[i] - out_o[i]))**2  #calculate error
		tst_mean_sqrd_error = (tst_mean_sqrd_error / len(tst_data))*100  # storing error
		tst_arctan_error = (tst_arctan_error / len(tst_data))*100  #storing error
		tst_rt_mean_sqrd_error = (tst_mean_sqrd_error ** (1.0 / 2))  #storing error
		return tst_mean_sqrd_error, tst_rt_mean_sqrd_error, tst_arctan_error

	# Split data in training and testing database
	def split_data(self):
		n_train_data = int(len(self.database)*0.8)  # 70% data for training
		n_testing_data = len(self.database) - n_train_data  #remaining percent for testing
		self.training_database = [[] for x in range(n_train_data)]  #Initialize
		self.testing_database = [[] for x in range(n_testing_data)]  #Initialize
		for i in range(n_train_data):
			self.training_database[i] = self.database[i]  # Training data
		for j in range(n_testing_data):
			self.testing_database[j] = self.database[n_testing_data+j]  #Testing data

	# Define accuracy of Neural Network at testing database mostly
	def accur(self, actual, predicted):
		for i in range(self.n_outputs):
			accuracy  = np.abs(actual[i] - predicted[i])  # add the differene between target and the prediction
		return (100 - ((accuracy / self.n_outputs)*100))  # divide with number to get the mean accuracy

	def train(self, store=False):
		# Starting the Machine Learning core algorithm
		h_weights, o_weights = list(self.initialize_network())  # initialize network
	
		# SPLIT DATABASE  ~~~> Training and Testing Database
		self.split_data()

		#STANDARDIZE TRAINING DATA
		tr_coll_means = self.col_mean(self.training_database)
		tr_stdevs = self.stdv_col(self.training_database, tr_coll_means)
		self.standardized_data(self.training_database, tr_coll_means, tr_stdevs)

		# Shuffle Data before training
		random.shuffle(self.training_database)
		random.shuffle(self.testing_database)

		# widgets for progreeebar in training period
		widgets = ['Training Network: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'), ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options
		pbar = ProgressBar(widgets=widgets, maxval=self.epoch)
		pbar.start()

		# Train Network
		for m in range(self.epoch):
			l_rate = self.dec_l_rate(m, self.init_l_rate)  # calculate new learning rate after decay
			for i in range(len(self.training_database)):
				for j in range(self.n_inputs):
					self.row[j] = float(self.training_database[i][j])  # create data for input from training database
				for j in range(self.n_outputs):
					self.target[j] = float(self.training_database[i][j + self.n_inputs])  # create table with output values
				for j in range(self.n_hidden):
					z = self.activate(h_weights[j], self.row)  # calculate activation of hidden layer
					self.h_out[j] = self.sigmoid(z)  # transfer with sigmoid
				for j in range(self.n_outputs):
					z = self.activate(o_weights[j], self.h_out)  # calculate activation of output layer with hidden layer output
					self.o_out[j] = self.sigmoid(z)  # transfer using sigmoid
				self.tr_accuracy += self.accur(self.target, self.o_out)  # Accuracy at every iteration
			# until this point we have the feed forward process and we start calculating the back propagation error
				for j in range(self.n_outputs):
					if loss == 'cost':
						self.o_cost[j] = self.cost(self.target[j], self.o_out[j])  # calculate output cost
					elif loss == 'crossentropy':
						self.o_cost[j] = self.cross_entropy(self.target[j], self.o_out[j])  # calculate output cost
				for j in range(self.n_outputs):
					if loss == 'cost':
						self.o_delta_cost[j] = self.d_cost(self.o_out[j], self.target[j])  # calculate the derivative cost of outputs
					elif loss =='crossentropy':
						self.o_delta_cost[j] = self.d_cross_entropy(self.o_out[j], self.target[j])  # calculate the derivative cost of outputs
					self.o_delta_sigmoid[j] = self.d_sigmoid(self.o_out[j])  # calculate the derivative sigmoid of outputs
				for j in range(self.n_outputs):
					for k in range(self.n_hidden):
						self.o_delta[j][k] = self.o_delta_cost[j] * self.o_delta_sigmoid[j] * self.h_out[k]  # calculate corrections for output weights
			# Up until this point we have calculated the corrections for the output nodes weights, we dont update the weights
			# until we calculate the corrections for the hidden layer ( the bias value do not change )
			# Calculating hidden layer cost for correction
				for j in range(self.n_hidden):
					for k in range(self.n_outputs):
						self.d_E_d_out_h[k] = self.o_delta_cost[k] * self.o_delta_sigmoid[k] * o_weights[k][j] # derivative of hidden with respect to the output
						self.final_d_E_d_out_h += self.d_E_d_out_h[k] # sum of each hidden node
					for l in range(self.n_inputs):
						self.h_delta[j][l] = self.final_d_E_d_out_h * self.d_sigmoid(self.h_out[j]) * self.row[l] # corrections for hidden
					self.final_d_E_d_out_h = 0
			# Up until this point we have calculated the corrections for the hidden nodes weights (bias doesnt change)
			# update weights and start again
				if optimize == 'sgd':
					for j in range(self.n_outputs):
						for k in range(self.n_hidden):
							self.u_o_weights[j][k] = o_weights[j][k] - (l_rate * self.o_delta[j][k])  #calculate new weights with SGD
					for j in range(self.n_hidden):
						for k in range(self.n_inputs):
							self.u_h_weights[j][k] = h_weights[j][k] - (l_rate * self.h_delta[j][k])  #calculate new weights with SGD
					for j in range(self.n_outputs):
						for k in range(self.n_hidden):
							o_weights[j][k] = self.u_o_weights[j][k]  #update weights
					for j in range(self.n_hidden):
						for k in range(self.n_inputs):
							h_weights[j][k] = self.u_h_weights[j][k]  #update weights
				elif optimize == 'adam':
					o_vdw, o_sdw, epsilon = self.Adam(self.o_delta, m, case="output")  # adam optimize for output layer
					h_vdw, h_sdw, epsilon = self.Adam(self.h_delta, m, case="hidden")  # adam optimize for hidden layer
					for j in range(self.n_outputs):
						for k in range(self.n_hidden):
							self.u_o_weights[j][k] = o_weights[j][k] - l_rate * ( (o_vdw[j][k]) / ( (o_sdw[j][k])**(0.5) + epsilon ) ) #calculate new weights based on Adam
					for j in range(self.n_hidden):
						for k in range(self.n_inputs):
							self.u_h_weights[j][k] = h_weights[j][k] - l_rate * ( (h_vdw[j][k]) / ( (h_sdw[j][k])**(0.5) + epsilon ) ) #calculate new weights based on Adam
					for j in range(self.n_outputs):
						for k in range(self.n_hidden):
							o_weights[j][k] = self.u_o_weights[j][k]  #update weights
					for j in range(self.n_hidden):
						for k in range(self.n_inputs):
							h_weights[j][k] = self.u_h_weights[j][k]  #update weights
			self.sum_tr_accuracy[m] = self.tr_accuracy / (len(self.training_database))  # Calculate the final mean accuracy after first epoch
			self.tr_accuracy = 0  # Clear accuracy for next epoch iterations
			self.tr_mean_sqrd_error, self.tr_rt_mean_sqrd_error, self.tr_arctan_error = self.tr_calc_err(o_weights, h_weights)  # calculate the error of the neural network after each iteration
			self.tr_counter += 1  # counter for storing training errors
		#	random.shuffle(training_database)  # shuffle training database for next epoch
			pbar.update(m)  # update progressbar
		pbar.finish() # finish progress

		# STORE FINAL WEIGHTS
		o_final_weights = o_weights
		h_final_weights = h_weights

		# Plots for Training
		x = np.arange(epoch)
		plt.subplot(2, 1, 1)
		plt.plot(x, self.tr_mean_sqrd_error, 'ro', label='Mean Square Error')
		plt.plot(x, self.tr_rt_mean_sqrd_error, 'b^', label='Root Mean Square Error')
		plt.plot(x, self.tr_arctan_error, 'gs', label='Arctan Error')
		plt.title('Training: Errors (Top) & Mean Accuracy (Bottom)', fontsize=15)
		plt.legend()
		plt.ylabel('Percentage (%)', fontsize=13)
		plt.subplot(2, 1, 2)
		plt.plot(x, self.sum_tr_accuracy, 'mx', label='Accuracy')
		plt.legend()
		plt.ylabel('Percentage (%)', fontsize=13)
		plt.xlabel('Epochs', fontsize=13)
		plt.show()

		#STANDARDIZE TESTING DATABASE
		tst_coll_means = self.col_mean(self.testing_database)
		tst_stdevs = self.stdv_col(self.testing_database, tst_coll_means)
		self.standardized_data(self.testing_database, tst_coll_means, tst_stdevs)

		# widgets for progreeebar in testing period
		widgets = ['Testing Network: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'), ' ', ETA(), ' ', FileTransferSpeed()] # see docs for other options
		pbar = ProgressBar(widgets=widgets, maxval=epoch)
		pbar.start()

		accuracy = [0 for x in range(len(self.training_database))]

		# Print to check for accuracy of predictions
		for i in range(len(self.training_database)):
			for j in range(self.n_inputs):
				self.row[j] = float(self.training_database[i][j])  # create data for input from training database
			for j in range(self.n_outputs):
				self.target[j] = float(self.training_database[i][j+self.n_inputs])  # create table with output values
			for j in range(self.n_hidden):
				z = self.activate(h_weights[j], self.row)  # calculate activation of hidden layer
				self.h_out[j] = self.sigmoid(z)  # transfer with sigmoid
			for j in range(self.n_outputs):
				z = self.activate(o_weights[j], self.h_out)  # calculate activation of output layer with hidden layer output
				self.o_out[j] = self.sigmoid(z)
			print('Iteration:', i)  #Print number of iteration
			print('Predicted:', self.o_out) # Print prediction of network
			print('Target:', self.target)  # Print target
			accuracy[i] = 100 - (abs(self.target[0] - self.o_out[0]))*100

		f = np.arange(len(accuracy))
		plt.plot(f, accuracy, 'ro')
		plt.show()

		# Testing Network with Testing Database
		for i in range(len(self.testing_database)):
			for j in range(self.n_inputs):
				self.row[j] = float(self.testing_database[i][j])  # create data for input from training database
			for j in range(self.n_outputs):
				self.target[j] = float(self.testing_database[i][j+self.n_inputs])  # create table with output values
			for j in range(self.n_hidden):
				z = self.activate(h_weights[j], self.row)  # calculate activation of hidden layer
				self.h_out[j] = self.sigmoid(z)  # transfer with sigmoid
			for j in range(self.n_outputs):
				z = self.activate(o_weights[j], self.h_out)  # calculate activation of output layer with hidden layer output
				self.o_out[j] = self.sigmoid(z)
			self.tst_accuracy += self.accur(self.target, self.o_out)  #Calculate accuracy at every prediction
			print('Iteration:', i)  #Print number of iteration
			print('Predicted:', self.o_out)  # Print prediction of network
			print('Target:', self.target)  # Print target
			pbar.update(m)  # update progressbar
		# Calculate the error of the neural network after each iteration
		tst_mean_sqrd_error, tst_rt_mean_sqrd_error, tst_arctan_error = self.tst_calc_err(o_weights, h_weights, self.testing_database)
		self.tst_accuracy = self.tst_accuracy / (len(self.testing_database))  # Divide to calculate mean accuracy
		pbar.finish() # finish progress

		# Plots for Testing
		x = np.arange(1)
		plt.subplot(2, 1, 1)
		plt.plot(x, tst_mean_sqrd_error, 'ro', label='Mean Square Error')
		plt.plot(x, tst_rt_mean_sqrd_error, 'b^', label='Root Mean Square Error')
		plt.plot(x, tst_arctan_error, 'gs', label='Arctan Error')
		plt.title('Testing: Errors (Top) & Mean Accuracy (Bottom)', fontsize=15)
		plt.ylabel('Percentage (%)', fontsize=13)
		plt.legend()
		plt.subplot(2, 1, 2)
		plt.plot(x, self.tst_accuracy, 'mx', label='Accuracy')
		plt.legend()
		plt.ylabel('Percentage (%)', fontsize=13)
		plt.xlabel('Epochs', fontsize=13)
		plt.show()
		
		if store:
			# store final weights after calculation
			save_as_array1 = np.asarray(o_final_weights)
			save_as_array2 = np.asarray(h_final_weights)
			np.save('Output Weights.npy', save_as_array1)
			np.save('Hidden Weights.npy', save_as_array2)


if __name__ == '__main__':
	from optparse import OptionParser

	parser = OptionParser()
	
	parser.add_option("-i", "--input", action='store', dest="number_inputs", help="Number of input nodes")
	parser.add_option("--i-hidden", "--hidden", action='store', dest="number_hidden", help="Number of hidden layer nodes")
	parser.add_option("-o", "--output", action="store", dest="number_output", help="Number of output nodes")
	parser.add_option("-d", "--database", action="store", dest="database", help="database file path")
	parser.add_option("-r", "--rate", action="store", dest="learning_rate", help="Initial learning rate")
	parser.add_option("-l", "--loss", action="store", dest="loss", help="loss function")
	parser.add_option("--o-optimize", "--optimize", action="store", dest="optimize", help="optimization")
	parser.add_option("-e", "--epochs", action="store", dest="epochs", help="number of epoches")

	(options, args) = parser.parse_args()

	# Load database
	if options.database:
		DATABASE = options.database
	else:
		DATABASE = 'Big_Database_2.csv'

	if options.number_inputs:
		n_inputs = int(options.number_inputs)
	else:
		n_inputs = 7

	if options.number_hidden:
		n_hidden = int(options.number_hidden)
	else:
		n_hidden = 7

	if options.number_output:
		n_outputs = int(options.number_output)
	else:
		n_outputs = 3

	if options.learning_rate:
		init_l_rate = int(options.learning_rate)
	else:
		init_l_rate = .9

	if options.epochs:
		epoch = int(options.epochs)
	else:
		epoch = 20

	if options.optimize:
		optimize = options.optimize
	else:
		optimize = 'sgd'

	if options.loss:
		loss = options.loss
	else:
		loss = 'cost'

	# Crucial Parameters to tune the accuracy of network
	# !!! Rule of thumb     !!!
	# !!! for small samples !!!
	# N_hidden = N_samples / (a*(N_inputs + N_outputs))   a = 2 - 10

	nn = NN(DATABASE, n_inputs, n_hidden, n_outputs, init_l_rate, epoch, optimize, loss)
	nn.load_csv()
	nn.train()

