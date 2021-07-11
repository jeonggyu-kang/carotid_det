from scipy import io
mat_file = io.loadmat ('../groundTruth.mat')
print (mat_file.keys())


for idx, key in enumerate(mat_file):
    print(mat_file[key])



print(type(mat_file['None']))