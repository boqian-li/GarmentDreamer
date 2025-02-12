import os


root1 = '/home/boqian/Desktop/exported_meshes_train'
root2 = '/home/boqian/Desktop/exported_codes_train'

id_list = []

for file in os.listdir(root1):
    if file.endswith('obj'):
        id = '_'.join(file.split('_')[:-1])
        id_list.append(id)

a = 0
for file in os.listdir(root2):
    if file.endswith('pt'):
        file_path = os.path.join(root2, file)
        id = file.split('.')[0]
        if not id in id_list:
            os.remove(file_path)
            print(id)
            a += 1

print(a)