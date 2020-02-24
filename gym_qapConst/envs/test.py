import numpy as np
import fis_generator as fisg
import os




def get_location_matrix(path,num_prod):
	prod_loc_matrix = np.zeros((num_prod,num_prod), int)
	file = open(path,"r")
	line = file.readline()
	for i in range(num_prod):
		k = [int(s) for s in line.split() if s.isdigit()]
		p1 = k[0]
		p2 = k[1]
		prod_loc_matrix[p1,p2] = 1
		line = file.readline()
	file.close()
	return prod_loc_matrix


path = os.getenv("HOME")+"/fisFolder/fisFile.txt"
matrix_fq = fisg.readFisFile(path)
num_prod = len(matrix_fq)
path = os.getenv("HOME")+"/prodLocFolder/prodLocFile"+str(num_prod)+".txt"
matrix_pl = get_location_matrix(path,num_prod)

matrix_dist = np.zeros((num_prod, num_prod), int)
for i in range(0,num_prod):
	for j in range(i+1,num_prod):
		matrix_dist[i,j] = matrix_dist[j,i] = j-i
for i in range(num_prod):
	matrix_dist[i,i] = i


def compute_mff_sum(matrix):
	diag = np.diag(matrix)
	diag.setflags(write=1)
	min_ind = np.argmin(diag,0)
	matrix_mff = matrix_pl[min_ind]
	diag[min_ind] = 90
	for i in range(1,num_prod):
		min_ind = np.argmin(diag,0)
		if diag[min_ind] == 90:
			break
		matrix_mff = np.vstack((matrix_mff,matrix_pl[min_ind]))
		diag[min_ind] = 90
	matrix_dp = np.dot(np.dot(matrix_mff,matrix_dist),np.transpose(matrix_mff))
	matrix_wd = matrix_dp*matrix_fq
	mff_sum = np.sum(matrix_wd)
	return mff_sum	

matrix_dp = np.dot(np.dot(matrix_pl,matrix_dist),np.transpose(matrix_pl))
matrix_wd = matrix_dp*matrix_fq
current_sum = np.sum(matrix_wd)
initial_sum = np.sum(matrix_wd)
mff_sum = compute_mff_sum(matrix_dp)
#print(initial_sum)
print(matrix_pl)