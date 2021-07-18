import csv
import json

def get_location_type(l_type):
    '''
        MidCCA 0
        Distal 1
        Bulb 2
        ICA 3
    '''
    try:
        ret = l_type.index('true')
    except:
        return -1
    return ret


def csv2dict(root):
    ret = {}
    '''
    ret = {
        file_image : {
            li : {'x': [], 'y': []}
            ma : {'x': [], 'y': []}
            location_type : int
            roi : []
        }
    }
    '''
    with open(root, 'r') as f:
        data = csv.reader(f)
        for idx, line in enumerate(data):
            if idx == 0:
                continue
            
            file_path = line[0]
            file_path = file_path.split('\\')[-1]
            roi = [ line[1], line[2], line[3], line[4] ]
            roi = list(map(int, roi))
            LI = {}
            MA = {}
            li_x = line[5] # LI - x
            li_y = line[6] # LI - y
            li_x = list(map(float, li_x.split(',')))
            li_y = list(map(float, li_y.split(',')))
            LI['x'] = li_x
            LI['y'] = li_y

            ma_x = line[7] 
            ma_y = line[8] 
            ma_x = list(map(float, ma_x.split(',')))
            ma_y = list(map(float, ma_y.split(',')))
            MA['x'] = ma_x
            MA['y'] = ma_y
            location_type = get_location_type(line[9:])
            if -1 == location_type:
                continue

            ret[file_path] = {}
            ret[file_path]['li'] = LI
            ret[file_path]['ma'] = MA
            ret[file_path]['location'] = location_type
            ret[file_path]['roi'] = roi          
    return ret        

if __name__ == '__main__':
    CSV_FILE_PATH = './gTruth.csv'
    json_data = csv2dict(CSV_FILE_PATH)

    with open('gTruth.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent='\t')