import csv
import json
f = open('/mnt/carotid_data/carotid.csv', 'r', encoding='utf-8')
data = csv.reader(f)
# 
file_path =[]
roi = [] # [1,4,6,8]
type1 = []
type2 = []
type3 = []
type4 = []
gt1_x = []
gt1_y = []
gt2_x = []
gt2_y = []
is_next_sample = True
for idx, line in enumerate(data):
    if idx == 0:
        continue
    if is_next_sample:
        file_name = line[0].split('\\')[-1]
        file_path.append(file_name[:-1])
        roi.append([line[1], line[2], line[3], line[4]])
        type1.append(line[5])
        type2.append(line[6])
        type3.append(line[7])
        type4.append(line[8])
    break

f.close()

json_data = {'file_name': file_path,
             'roi': roi,
             'type1': type1,
             'type2': type2,
             'type3': type3,
             'type4': type4}
with open('carotid.json', 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent='\t')
