#!/usr/bin/env
import numpy as np
from PrioritizedSumTree import PrioritizedSumTree

#TODO - implement Tiered memories similar to cache levels in a processor
#TODO - try out some of the smarter replay choosing methods on my scribbles of junk

class SarstReplayMemory:
	"""This memory holds three numpy arrays, each of which store the state, policy, value
	It works by sampling a batch of corresponding indexes
	from each of the four (s, p, v) arrays. This sampling is done by using a prioritized sum tree, which will pick
	samples that have high rewards associated with them under the assumption that they are more important.
	"""
	def __init__(self, capacity, state_size, policy_size, useLSTM, trace_length, prioritize=False, priority_epsilon=0.001, priority_alpha=0.5):
		"""
		Args:
			Capacity - int - How many samples to hold in the array. Samples all exist for exactly the same amount of time
			State_Size - numpy array shape tuple -  specifying the dimensions of the input
			prioritize - Boolean, whether to use the prioritized sum tree or to just randomly pick
			priority epsilon - float - small number. the smoothing param to add to the reward priority calculation if priortize is true
			priority alpha - float - between 0-1, the exponent used in the priority calculatiopn if prioritize is true
		"""
		# Some book-keeping so we know how big the memory is
		self.memory_capacity = capacity #max size
		self.memory_size = 0
		self.memory_pointer_index = 0
		
		self.useLSTM = useLSTM
		self.trace_length = trace_length

		self.prioritize = prioritize
		self.total_priority_sum = 0.
		if prioritize:
			self.priority_epsilon = priority_epsilon
			self.priority_alpha = priority_alpha
			self.priority_tree = PrioritizedSumTree(capacity)
		
		# Create the SARS memory
		self.state_size = state_size
		new_state_size = [self.memory_capacity] + [s for s in state_size]
		self.states = np.zeros(shape=new_state_size, dtype=np.int8)
		new_policy_size = [self.memory_capacity] + [s for s in policy_size]
		self.policies = np.zeros(shape=new_policy_size, dtype=np.uint16)
		self.values = np.zeros(shape=(self.memory_capacity), dtype=np.float32)


	def save_memory(self, memoryFile):
		np.savez(memoryFile, 
			states = self.states, 
			policies = self.policies,
			values = self.values,
			memory_size = self.memory_size,
			memory_pointer_index = self.memory_pointer_index,
			useLSTM = self.useLSTM,
			trace_length = self.trace_length)
		if self.prioritize:
			self.priority_tree.save_sumTree()			
		print("=== Memory saved as \"{}.npz\" ===".format(memoryFile))
	
	def restore_memory(self, memoryFile):
		npzFile = np.load("{}.npz".format(memoryFile))
		self.states = npzFile['states']
		self.policies = npzFile['policies']
		self.values = npzFile['values']		
		self.memory_size = npzFile['memory_size']		
		self.memory_pointer_index = npzFile['memory_pointer_index']
		self.useLSTM = npzFile['useLSTM']
		self.trace_length = npzFile['trace_length']
		if self.prioritize:
			self.priority_tree.restore_sumTree()				
		print("=== Memory restored from \"{}.npz\" ===".format(memoryFile))
		
		
	def add_to_memory(self, state, policy, value, priority):
		"""
		Adds new value to s, p, v arrays when new observation is made.
		Write all the information to the current pointer in memory
		"""
		self.states[self.memory_pointer_index] = state
		self.policies[self.memory_pointer_index] = policy
		self.values[self.memory_pointer_index] = value

		# if using a prioritized memory, need to add it the sample to that as well
		if self.prioritize:	
			# calculates a priority value, and adds it to the tree. Note we NEED absolute value
			# for the correctness of the sum tree
			sample_priority = (abs(priority) + self.priority_epsilon)**self.priority_alpha
			self.priority_tree.add(sample_priority)

		# we have to increment the memory pointer index so we write to the correct spot
		# note that the prioritized sum tree does this on its own, which is why we dont have it explicitly done above
		self.memory_pointer_index = (self.memory_pointer_index + 1) % self.memory_capacity

		# also update the memory size as a safety check against batches that are larger than the memory size
		self.memory_size = min(self.memory_size+1, self.memory_capacity)


	def get_batch_sample(self, batch_size):
		"""
		Returns a numpy array of batch_size samples complete with s, p, v. This is to be fed into a neural network
		such that the network can compute a one-hot dot product with the action that it cares about to see how
		accurate the network is
		"""
		if self.memory_size < batch_size * self.trace_length:
			# TODO - might be a better idea to just keep sampling from them anyway and repeat samples
			raise ValueError("Cannot read a batch of %d samples when memory only has %d samples stored" % (batch_size, self.memory_size))

		chosen_sarst_indexes = []

		# either use the prioritized sum tree to return samples based on their reward values so that more important ones
		# are chosen, or do it randomly
		if self.prioritize:
			while len(chosen_sarst_indexes) < batch_size:
				chosen_idx, relative_priority = self.priority_tree.get()
				#print("chose replay idx %d with priority %.4f" % (chosen_idx, relative_priority))
				
				# the tree starts at 0, therefore the leaves start at half of the tree size
				# we need to convert to the index of the sarst memory by subtracting the capcacity+1
				chosen_idx -= self.memory_capacity-1
				#print("this chosen reward should read %.4f\n" % (abs(self.rewards[chosen_idx]) + self.priority_epsilon)**self.priority_alpha)
				chosen_sarst_indexes.append(chosen_idx)
				
			"""print("{:.0f} | {:.0f} | {:.0f} | {:.0f}".format(np.mean(chosen_sarst_indexes),
											np.std(chosen_sarst_indexes),
											np.min(chosen_sarst_indexes),
											np.max(chosen_sarst_indexes)))"""
		else:		
			# choose randomly
			while len(chosen_sarst_indexes) < batch_size:
				chosen_sarst_indexes.append(np.random.randint(low=0, high=self.memory_size))

		if self.useLSTM:
			states = []
			policies = []
			values = []
			
			for index in chosen_sarst_indexes:
				index += 1 # to include index 
				for i in range(index-self.trace_length,index):
				#for i in range(index,index+self.trace_length):
					states.append(self.states[i])
					policies.append(self.policies[i])
					values.append(self.values[i])
			
			# Reshape
			state_shape = [batch_size*self.trace_length] + [s for s in self.state_size]
			states = np.reshape(np.array(states),state_shape)
			policies = np.reshape(np.array(policies),[batch_size*self.trace_length,])
			values = np.reshape(np.array(values),[batch_size*self.trace_length,])
		else:
			states = self.states[chosen_sarst_indexes]
			policies = self.policies[chosen_sarst_indexes]
			values = self.values[chosen_sarst_indexes]
		
		return states, policies, values
   
