output_file_0 = '/media/vipsl04/Harddisk/output_label_0.txt'
output_file_1 = '/media/vipsl04/Harddisk/output_label_1.txt'
result_file = '/media/vipsl04/Harddisk/average_confidence1.txt'
# 定义一个函数来统计predicted label为0的数量
def count_predicted_label_0(file_path):
    count = 0
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # 查找predicted label的部分
            if 'predicted label: 0' in line:
                count += 1
    return count
def adjust_confidence(file_path):
    updated_lines = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # 如果predicted label是0，则调整confidence值
            if 'predicted label: 1' in line:
                # 找到confidence的位置，更新confidence
                parts = line.strip().split(',')
                confidence_part = [part for part in parts if 'confidence' in part][0]
                confidence_value = float(confidence_part.split(':')[1].strip())
                new_confidence = 1 - confidence_value
                # 替换原有的confidence值
                updated_line = line.replace(str(confidence_value), str(new_confidence))
                updated_lines.append(updated_line)
            else:
                updated_lines.append(line)

    # 将更新后的内容写入文件
    with open(file_path, 'w') as f:
        f.writelines(updated_lines)


def calculate_average_confidence(file_path, result_file):
    confidence_values = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            # 提取confidence值
            if 'confidence' in line:
                parts = line.strip().split(',')
                confidence_part = [part for part in parts if 'confidence' in part][0]
                confidence_value = float(confidence_part.split(':')[1].strip())
                confidence_values.append(confidence_value)
    averages = []
    for i in range(0, len(confidence_values), 16):
        group = confidence_values[i:i + 16]
        if len(group) != 16:
            print(1)
        avg_confidence = sum(group) / 16
        averages.append(avg_confidence)

    # 将结果写入新文件
    with open(result_file, 'w') as f:
        for avg in averages:
            f.write(f"{avg}\n")
# 统计并输出结果
#count_0 = count_predicted_label_0(output_file_0)
#count_1 = count_predicted_label_0(output_file_1)
adjust_confidence(output_file_0)
#print(f"In output_label_0.txt, predicted label 0 count: {count_0}")
#print(f"In output_label_1.txt, predicted label 0 count: {count_1}")
calculate_average_confidence(output_file_0, result_file)
#calculate_average_confidence(output_file_0, result_file)