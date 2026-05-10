# 定义输入和输出文件路径
input_file = '/media/vipsl04/Harddisk/output4.txt'
output_file_0 = '/media/vipsl04/Harddisk/output_label_0.txt'
output_file_1 = '/media/vipsl04/Harddisk/output_label_1.txt'

# 打开输入文件进行读取
with open(input_file, 'r') as f:
    lines = f.readlines()

# 打开输出文件进行写入
with open(output_file_0, 'w') as f_0, open(output_file_1, 'w') as f_1:
    for line in lines:
        # 解析每一行数据
        parts = line.strip().split(',')
        true_label = int(parts[0].split(':')[1].strip())  # 获取True label的值

        # 根据True label的值将行写入相应的文件
        if true_label == 0:
            f_0.write(line)
        elif true_label == 1:
            f_1.write(line)