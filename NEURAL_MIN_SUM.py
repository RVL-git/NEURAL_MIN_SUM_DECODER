#входные значения
train= True
SPA = False
MN = not SPA
zeros_train= True 
zeros_testing = False
no_sigma_train = False
no_sigma_test = False


import numpy as np
import tensorflow.compat.v1 as tf
import sys
from tensorflow.python.framework import ops
import os
import pylab
  
tf.disable_v2_behavior()


#описание методов и параметров кода используемого в модели
class Code:
    def __init__(self):
        self.num_edges = 0

#загрузка параметров кода
def load_code(H_filename, G_filename):
	
	with open(H_filename) as f:
		n,m = [int(s) for s in f.readline().split(' ')]
		k = n-m

		var_degs = np.zeros(n).astype(np.int) # cтепень каждой переменной вершины
		chk_degs = np.zeros(m).astype(np.int) # степень каждой проверочной вершины

        #инициализация проверочной матрицы H
		H = np.zeros([m,n]).astype(np.int)
		max_var_degsree, max_chk_degree = [int(s) for s in f.readline().split(' ')]
		f.readline() 
		f.readline()

        #определение проверочной матрицы H
		var_edges = [[] for _ in range(0,n)]
		for i in range(0,n):
			row_string = f.readline().split(' ')
			var_edges[i] = [(int(s)-1) for s in row_string[:-1]]
			var_degs[i] = len(var_edges[i])
			H[var_edges[i], i] = 1

		chk_edges = [[] for _ in range(0,m)]
		for i in range(0,m):
			row_string = f.readline().split(' ')
			chk_edges[i] = [(int(s)-1) for s in row_string[:-1]]
			chk_degs[i] = len(chk_edges[i])

		d = [[] for _ in range(0,n)]
		edge = 0
		for i in range(0,n):
			for j in range(0,var_degs[i]):
				d[i].append(edge)
				edge += 1

		u = [[] for _ in range(0,m)]
		edge = 0
		for i in range(0,m):
			for j in range(0,chk_degs[i]):
				v = chk_edges[i][j]
				for e in range(0,var_degs[v]):
					if (i == var_edges[v][e]):
						u[i].append(d[v][e])

		num_edges = H.sum()

	if G_filename == "":
		G = []
	else:
		if "BCH" in H_filename: 
			G = np.loadtxt(G_filename).astype(np.int)
			G = G.transpose()
		else:
			P = np.loadtxt(G_filename,skiprows=2)
			G = np.vstack([P.transpose(), np.eye(k)]).astype(np.int)

	code = Code()
	code.H = H
	code.G = G
	code.var_degs = var_degs
	code.chk_degs = chk_degs
	code.num_edges = num_edges
	code.u = u
	code.d = d
	code.n = n
	code.m = m
	code.k = k
	return code


#вычисление синдрома 
def syndrome(soft_output, code):
	H = code.H
	G = code.G
	n = code.n
	m = code.m
	k = code.k
	soft_syndrome = []
	for c in range(0, m): 
		variable_nodes = []
		for v in range(0, n):
			if H[c,v] == 1: variable_nodes.append(v)
		temp = tf.gather(soft_output,variable_nodes)
		temp1 = tf.reduce_prod(tf.sign(temp),0)
		temp2 = tf.reduce_min(tf.abs(temp),0)
		soft_syndrome.append(temp1 * temp2)
	soft_syndrome = tf.stack(soft_syndrome)
	return soft_syndrome



#параметры модели при обучении и тестировании
seed = 786000 #рекомендованный Hacmhani генератор
np.random.seed(seed)
snr_lo = 1.0
snr_hi = 8.0
snr_step = 1.0
min_frame_errors = 100
max_frames = 1000000
num_iterations = 5
H_filename = ""
G_filename = ""
output_filename = ""
L = 0.5
steps = 20001
provided_decoder_type = ""


#полученные из файла параметры кода
if zeros_testing: G_filename = ""
code = load_code(H_filename, G_filename)
H = code.H
G = code.G
var_degs = code.var_degs
chk_degs = code.chk_degs
num_edges = code.num_edges
u = code.u
d = code.d
n = code.n
m = code.m
k = code.k

batch_size = 120
train_ds = tf.placeholder(tf.float32, shape=(n,batch_size))
train_ls= tf.placeholder(tf.float32, shape=(n,batch_size))





#МОДЕЛЬ И МЕТОДЫ ДЕКОДЕРА
class Decoder:
	def __init__(self, decoder_type, random_seed, learning_rate):
		self.decoder_type = decoder_type
		self.random_seed = random_seed
		self.learning_rate = learning_rate

# Вычисление сообщения от проверочной вершины к переменной
def v2c(cv, iteration, soft_input):
	weighted_soft_input = soft_input
	
	edges = []
	for i in range(0, n):
		for j in range(0, var_degs[i]):
			edges.append(i)
	reordered_soft_input = tf.gather(weighted_soft_input, edges)
	
	vc = []
	edge_order = []
	for i in range(0, n): 
		for j in range(0, var_degs[i]):
			edge_order.append(d[i][j])
			extrinsic_edges = []
			for jj in range(0, var_degs[i]):
				if jj != j: 
					extrinsic_edges.append(d[i][jj])

			if extrinsic_edges:
				temp = tf.gather(cv,extrinsic_edges)
				temp = tf.reduce_sum(temp,0)
			else:
				temp = tf.zeros([batch_size])
			if SPA: temp = tf.cast(temp, tf.float32)
			vc.append(temp)
	
	vc = tf.stack(vc)
	new_order = np.zeros(num_edges).astype(np.int)
	new_order[edge_order] = np.array(range(0,num_edges)).astype(np.int)
	vc = tf.gather(vc,new_order)
	vc = vc + reordered_soft_input
	return vc

# вычисление сообщения от переменной вершины к проверочной
def c2v(vc, iteration):
	cv_list = []
	prod_list = []
	min_list = []
	
	if SPA:
		vc = tf.clip_by_value(vc, -10, 10)
		tanh_vc = tf.tanh(vc / 2.0)
	edge_order = []
	for i in range(0, m): 
		for j in range(0, chk_degs[i]):
			edge_order.append(u[i][j])
			extrinsic_edges = []
			for jj in range(0, chk_degs[i]):
				if jj != j:
					extrinsic_edges.append(u[i][jj])
			if SPA:
				temp = tf.gather(tanh_vc,extrinsic_edges)
				temp = tf.reduce_prod(temp,0)
				temp = tf.log((1+temp)/(1-temp))
				cv_list.append(temp)
			if MN:
				temp = tf.gather(vc,extrinsic_edges)
				temp1 = tf.reduce_prod(tf.sign(temp),0)
				temp2 = tf.reduce_min(tf.abs(temp),0)
				prod_list.append(temp1)
				min_list.append(temp2)
	
	if SPA:
		cv = tf.stack(cv_list)
	if MN:
		prods = tf.stack(prod_list)
		mins = tf.stack(min_list)
		if decoder.decoder_type == "NOMS":
			offsets = tf.nn.softplus(decoder.B_cv[iteration])
			mins = tf.nn.relu(mins - tf.tile(tf.reshape(offsets,[-1,1]),[1,batch_size]))
		cv = prods * mins
	
	new_order = np.zeros(num_edges).astype(np.int)
	new_order[edge_order] = np.array(range(0,num_edges)).astype(np.int)
	cv = tf.gather(cv,new_order)

	return cv

# получение апостериорных значений LLR
def marginalize(soft_input, iteration, cv):
	weighted_soft_input = soft_input

	soft_output = []
	for i in range(0,n):
		edges = []
		for e in range(0,var_degs[i]):
			edges.append(d[i][e])

		temp = tf.gather(cv,edges)
		temp = tf.reduce_sum(temp,0)
		soft_output.append(temp)

	soft_output = tf.stack(soft_output)

	soft_output = weighted_soft_input + soft_output
	return soft_output


#одна итерация BP. использутся при построении графа Таннера
def BP_iter(soft_input, soft_output, iteration, cv, m_t, loss, labels):

	vc = v2c(cv,iteration,soft_input)

	cv = c2v(vc,iteration)

	# выход для итерации
	soft_output = marginalize(soft_input, iteration, cv)
	iteration += 1

	# L = 0.5
	CE_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels)) / num_iterations
	syndrome_loss = tf.reduce_mean(tf.maximum(1. - syndrome(soft_output, code),0) ) / num_iterations
	new_loss = L * CE_loss + (1-L) * syndrome_loss
	loss = loss + new_loss

	return soft_input, soft_output, iteration, cv, m_t, loss, labels

#пока условие выполняется при построениии графа
def continue_condition(soft_input, soft_output, iteration, cv, m_t, loss, labels):
	condition = (iteration < num_iterations)
	return condition

# построение графа Таннера (используется как сетка на которой мы будем обучать)
def get_graph(soft_input, labels):
	return tf.while_loop(
		continue_condition, # iteration < max iteration?
		BP_iter, # compute messages for this iteration
		[
			soft_input, # soft input for this iteration
			soft_input,  # soft output for this iteration
			tf.constant(0,dtype=tf.int32), # iteration number
			tf.zeros([num_edges,batch_size],dtype=tf.float32), # cv
			tf.zeros([num_edges,batch_size],dtype=tf.float32), # m_t
			tf.constant(0.0,dtype=tf.float32), # loss
			labels
		]
		)


global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
learning_rate = starter_learning_rate
decoder = Decoder(decoder_type=provided_decoder_type, random_seed=1, learning_rate = learning_rate)

if decoder.decoder_type == "NOMS":
		decoder.B_cv = tf.Variable(tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0))#tf.Variable(1.0 + tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0))#tf.Variable(1.0 + tf.truncated_normal([num_iterations, num_edges],dtype=tf.float32,stddev=1.0/num_edges))


#Сессия TF		
with tf.Session(config=config) as session: 
	# Симулируем каждые значения SNR
	SNRs = np.arange(snr_lo, snr_hi+snr_step, snr_step)
	if (batch_size % len(SNRs)) != 0: print("error")

	BERs = []
	SERs = []
	FERs = []
    #строим граф
	print("\nBuilding the decoder graph...")
	belief_propagation = get_graph(soft_input=train_ds, labels=tf_train_labels)
	if TRAINING:
		training_loss = belief_propagation[5]
		loss = training_loss
        #оптимизатор Адам
		optimizer = tf.train.AdamOptimizer(learning_rate=decoder.learning_rate).minimize(loss,global_step=global_step)
	print("Suc.\n")
	init = tf.global_variables_initializer()

	covariance_matrix = np.eye(n)
	eta = 0.99
	for i in range(0,n):
		for j in range(0,n):
			covariance_matrix[i,j] = eta**np.abs(i-j)

	session.run(init)
	
	if TRAINING:
		step = 0
		while step < steps:
			# генеруем TRAIN SET
			if not zeros_trainING:
				# генерация сообщений
				messages = np.random.randint(0,2,[k,batch_size])

				# кодирование сообщений
				codewords = np.dot(G, messages) % 2
				
				# модуляция кодововых слов
				BPSK_codewords = (0.5 - codewords.astype(np.float32)) * 2.0

                #конечный выход
				soft_input = np.zeros_like(BPSK_codewords)
				channel_information = np.zeros_like(BPSK_codewords)
			else:#аналогичный алгоритм, но все кодовые слова нулевые
				codewords = np.zeros([n,batch_size])
				BPSK_codewords = np.ones([n,batch_size])
				soft_input = np.zeros_like(BPSK_codewords)
				channel_information = np.zeros_like(BPSK_codewords)

			# создаем батч с различными значениями SNR, добавляем шум к нашему выходу из канала
			for i in range(0,len(SNRs)):
				sigma = np.sqrt(1. / (2 * (np.float(k)/np.float(n)) * 10**(SNRs[i]/10)))
				noise = sigma * np.random.randn(n,batch_size//len(SNRs))
				start_idx = batch_size*i//len(SNRs)
				end_idx = batch_size*(i+1)//len(SNRs)
				channel_information[:,start_idx:end_idx] = BPSK_codewords[:,start_idx:end_idx] + noise
                # конвертиурем в LLR формат
				if no_sigma_train:
					soft_input[:,start_idx:end_idx] = channel_information[:,start_idx:end_idx]
				else:
					soft_input[:,start_idx:end_idx] = 2.0*channel_information[:,start_idx:end_idx]/(sigma*sigma)


			# скармливаем батч и обучаем смещения
			batch_data = soft_input
			batch_labels = codewords
			feed_dict = {train_ds : batch_data, train_ls: batch_labels}
			[_] = session.run([optimizer], feed_dict=feed_dict) 

			if decoder.relaxed and TRAINING: 
				print(session.run(R))

			if step % 100 == 0:
				print(str(step) + " minibatches completed")

			step += 1
		
		print("Trained decoder on " + str(step) + " minibatches.\n")

	# testing phase
	print("Testing...")
	for SNR in SNRs:
		# Симулируем каждые значения SNR
		sigma = np.sqrt(1. / (2 * (np.float(k)/np.float(n)) * 10**(SNR/10)))
		frame_count = 0
		bit_errors = 0

		# по фреймам
		while ((FE < min_frame_errors) or (frame_count < 100000)) and (frame_count < max_frames):
			frame_count += batch_size

			if not zeros_testing:
				#генерируем сообщение
				messages = np.random.randint(0,2,[batch_size,k])

				# кодируем сообщение
				codewords = np.dot(G, messages.transpose()) % 2

				# модулируем кодовое слово
				BPSK_codewords = (0.5 - codewords.astype(np.float32)) * 2.0

			# добавляем белый гауссовский шум 
			noise = sigma * np.random.randn(BPSK_codewords.shape[0],BPSK_codewords.shape[1])
			channel_information = BPSK_codewords + noise

			# конвертиурем в LLR формат
			if no_sigma_test:
				soft_input = channel_information
			else:
				soft_input = 2.0*channel_information/(sigma*sigma)

			#запускаем BP 
			batch_data = soft_input
			feed_dict = {train_ds : batch_data, train_ls: codewords}
			soft_outputs = session.run([belief_propagation], feed_dict=feed_dict)
			soft_output = np.array(soft_outputs[0][1])
			recovered_codewords = (soft_output < 0).astype(int)

			# Пересчитываем кол-во ошибок
			errors = codewords != recovered_codewords
			bit_errors += errors.sum()
			frame_errors += (errors.sum(0) > 0).sum()

			FE = frame_errors

		
		print("SNR: " + str(SNR))

		bit_count = frame_count * n
		BER = np.float(bit_errors) / np.float(bit_count)
		BERs.append(BER)
		print("BER: " + str(BER))


	# суммарно
	print("BERs:")
	print(BERs)
    offset = session.run(decoder.B_cv)

    #отрисовываем график
    figure = pylab.figure()
	axes = figure.add_subplot (1, 1, 1)
	pylab.plot(np.arange(snr_lo,snr_hi + 1), BERs)
	pylab.grid()
	#ax.grid()
	axes.set_yscale ('log')

	pylab.show()
	