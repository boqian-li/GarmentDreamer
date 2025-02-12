import os

root = "/home/boqian/Desktop/exported_meshes_train/"

file_ids = []
with open(os.path.join(root, 'bad.txt'), 'r') as file:
    lines = file.readlines()

    for line in lines:
        # 去除行尾的换行符并按空格分割字符串
        parts = line.strip().split(':')

        # 获取文件名和值
        filename = parts[0].strip()  # 获取文件名部分并去除首尾空格
        value = float(parts[1].strip())  # 获取值部分并转换为浮点数

        file_id = filename.split('.')[0]
        file_ids.append(file_id)

print(file_ids)
print(len(file_ids))

a = 0
for file in os.listdir(root):
    if file.endswith('obj'):
        file_path = os.path.join(root, file)
        id = '_'.join(file.split('_')[:-1])
        if id in file_ids or id.startswith('skirt_8_panels'):
            os.remove(file_path)
            print(id)
            a += 1

print(a)