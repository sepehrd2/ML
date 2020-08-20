
import torch
import torch.nn as nn
from   torch.autograd import Variable
import numpy    as np
import pandas
import models   as my_model
from   scipy          import stats
############################################################
# reading the training file
# getting X and Y

print ("Initializing some paramteres ... ")
csvtrain    = pandas.read_csv('./cs446-fa19/train.csv')
cvstest     = pandas.read_csv('./cs446-fa19/test.csv')

N_train       = csvtrain.values.shape[0]
N_test        = cvstest.values.shape[0]
N_columns     = csvtrain.values.shape[1] - 12
Y_training    = np.zeros((N_train, 4))
X_training    = np.zeros((N_train, N_columns))
X_test        = np.zeros((N_test,  N_columns))
ini_par_train = np.zeros((N_train, 3))
ini_par_test  = np.zeros((N_test, 3))

whichmodel = "deepNN_1"

#assigning labels
print ("Reading labels ... ")
for j in range(0, 4):
	Y_training[:,j] =  csvtrain.values[:,j + 9]

#assigning input data
print ("Reading input data ... ")
for j in range(0 , N_columns - 5):
	X_training[:,j] = stats.zscore(csvtrain.values[:,j + 17])

#assigning test data
print ("Reading test data ... ")
for j in range(0 , N_columns - 5):
	X_test[:,j]     = stats.zscore(cvstest.values[:,j + 13])

X_training[:,N_columns - 4] = stats.zscore(csvtrain.values[:,6])
X_training[:,N_columns - 5] = stats.zscore(csvtrain.values[:,7])

X_test[:,N_columns - 4] = stats.zscore(cvstest.values[:,6])
X_test[:,N_columns - 5] = stats.zscore(cvstest.values[:,7])

inputDim     = N_columns                       # takes variable 'x' 
outputDim    = 4                               # takes variable 'y'
learningRate = 0.01 
epochs       = 8000

print ("Reading ini parameters ... ")

for i in range(0, N_train):
	if i != 1185:
		for j in range(0, 3):
			a = csvtrain.values[i][14 + j].strip('[')
			ini_par_train[i][j] = np.sqrt(np.dot(np.fromstring(a.strip(']'), dtype = float, sep = ','), np.fromstring(a.strip(']'), dtype = float, sep = ',')))
	if i % 100 == 0:
		print('Reading data from the training data set {} out of {}'.format(i, N_train))

for i in range(0, N_test):
	for j in range(0, 3):
		a = cvstest.values[i][10 + j].strip('[')
		ini_par_test[i][j] = np.sqrt(np.dot(np.fromstring(a.strip(']'), dtype = float, sep = ','), np.fromstring(a.strip(']'), dtype = float, sep = ',')))
	if i % 100 == 0:
		print('Reading data from the test data set {} out of {}'.format(i, N_test))

for j in range(N_columns - 3, N_columns):
	X_training[:,j] = stats.zscore(ini_par_train[:,j - N_columns + 3])

for j in range(N_columns - 3, N_columns):
	X_test[:,j] = stats.zscore(ini_par_test[:,j - N_columns + 3])

print ("Start the training ... ")

##Choosing loss and gradeinet methods and model
if whichmodel == "linearRegression":
	model    = my_model.linearRegression(inputDim, outputDim)
elif whichmodel == "deepNN":
	model    = my_model.deepNN(inputDim, outputDim)
elif whichmodel == "deepNN_1":
	model    = my_model.deepNN_1(inputDim, outputDim)

criterion = nn.SmoothL1Loss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate, amsgrad=True)

# Converting inputs and labels to Variable
inputs = Variable(torch.from_numpy(X_training).float())
labels = Variable(torch.from_numpy(Y_training).float())

#Training the model
for epoch in range(epochs):

	optimizer.zero_grad()
	# get output from the model, given the inputs
	outputs = model(inputs)
	# get loss for the predicted output
	loss = criterion(outputs, labels)
	# get gradients w.r.t to parameters
	loss.backward()
	# update parameters
	optimizer.step()	
	if epoch % 100 == 0:
		print('epoch {}, loss {}'.format(epoch, loss.item()))

#Testing 
with torch.no_grad(): 
	predicted = model(Variable(torch.from_numpy(X_test)).float()).data.numpy()

#outputing the results

csv_file = open('prediction.csv', 'w')
csv_file.write('id,Predicted\n')

for i in range(0, N_test):
	csv_file.write('test_' + str(i) + '_val_error,')
	csv_file.write('{}\n'.format(predicted[i][0]))
	csv_file.write('test_' + str(i) + '_train_error,')
	csv_file.write('{}\n'.format(predicted[i][2]))
