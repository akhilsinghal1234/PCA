import numpy as np
import matplotlib.pyplot as plt
import glob,numpy
from numpy import linalg as LA

color = ['tomato','greenyellow','navy','hotpink','skyblue','khaki','lightcoral']
# colors = ['r','b','purple','k']
reduced_dim, original_dim = 0,0

classes = ["industrial_area","patio","aqueduct"]
data,cov,eigen_val,eigen_vec= [],[],[],[]
for i in range(len(classes)):
	t = []
	data.append(t)

i = 0
for class_ in classes: 
	files = glob.glob("Train/" + class_ + "/*.txt")	# "Test/" + 
	for file in files:
		fp = open(file,"r")
		for line in fp:
			t = []
			for s in line.split():
				t.append(int(s))
		data[i].append(t)
	i += 1
original_dim = len(data[0][0])
all_data = []

# Concatenate all data

for i in range(len(data)):
	for data_ in data[i]:
		all_data.append(data_)

cov = (numpy.cov(m = all_data,rowvar = False))

#  Find eigen vals and vectors

eigenvalues, eigenvectors = LA.eig(cov)
eigen_val = eigenvalues
eigen_vec = eigenvectors

# Plot graphs for eigen values
plt.figure()
eig_vals = eigen_val
re, ima = [],[]
for val in eig_vals:
	re.append(val.real)
	ima.append(val.imag)
plt.scatter(re,ima,marker = '+', c = 'r')
plt.savefig("Eigen-values.png")

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_vals = eigen_val
for j in range(len(eigen_val)):
	for k in range(len(eigen_val)-1):
		if eigen_val[k] < eigen_val[k+1]:
			eigen_val[k], eigen_val[k + 1] = eigen_val[k + 1], eigen_val[k]
			eigen_vec[:,k], eigen_vec[:,k + 1] = eigen_vec[:,k + 1], eigen_vec[:,k]


# Explained variance
plt.figure()

eig_vals = eigen_val
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in eig_vals]
cum_var_exp = np.cumsum(var_exp)
indexes = list(range(1,(len(eig_vals)+1)))
plt.ylim((0,120))
plt.xlim((0,64))
plt.xlabel('Components used')
plt.ylabel('cummulative variance explained')
plt.plot(indexes,cum_var_exp,label = 'reduced dimension = 42',marker = '+', c= color[0])
plt.axvline(x=42,c = 'k')
plt.axhline(y=100,c = 'k')
plt.legend()
plt.savefig("Var_explained.png")

# dimension which explains most of the variance can be used here, specific to data being used(see the graph formed)
reduced_dim = 38 

red_mat = numpy.zeros(shape = (original_dim,reduced_dim))	# ,dtype=complex

for i in range(reduced_dim):
	for j in range(original_dim):
		red_mat[j][i] = eigen_vec[:,i][j].real

mat = red_mat.T

# Writes new feature vectors in Train folder with suffix as _red to folders
for i in range(len(classes)):
	file,counter = '',0
	for t in data[i]:
		file = "Train/" + classes[i] + "_red/"			# Test/
		counter += 1
		file = file + str(counter) + ".txt"
		fp = open(file,"w")
		new_d = np.dot(mat,t)
		for item in new_d:
			fp.write("%s " % item)
		fp.close()