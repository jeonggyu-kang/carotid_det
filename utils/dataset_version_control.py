import os
import json
import shutil
import glob
from easydict import EasyDict

import cv2

class DVC:
    def __init__(self, src_root_dir, dst_root_dir, old_json, dst_json, thres=30, vis=None):
        '''
            args
            src_root_dir: str
                새로운 데이터셋의 루트 디렉터리
            dst_root_dir: str
                기존 데이터셋 루트 디렉터리
        '''
        self.src_root_dir = src_root_dir
        self.dst_root_dir = dst_root_dir
        self.old_json = old_json
        self.dst_json = dst_json
        self.vis = vis
        if self.vis:
            os.makedirs(self.vis, exist_ok=True)


        self.new_dataset_start_index = self._count_existing_samples()

        self.roi_expand_thres = thres

        os.makedirs(self.dst_root_dir, exist_ok=True)
            
    def _count_existing_samples(self):
        samples = []
        samples += glob.glob(self.dst_root_dir + '/.bmp')
        samples += glob.glob(self.dst_root_dir + '/.BMP')
        return len(samples)
            
    def _get_json_list(self):
        src_files = []
        
        for (root, dirs, files) in os.walk(self.src_root_dir):
            if len(files) > 0:
                for file_name in files:
                    if file_name.split('.')[-1].lower() == 'json':
                        
                        json_abs_path = os.path.join(root, file_name)
                        bmp_abs_path = json_abs_path.replace('.json', '.bmp')

                        src_files.append((json_abs_path, bmp_abs_path))
                        
        return src_files 

    def _get_sample_start_index(self):
        return self.new_dataset_start_index


    def _get_new_dict(self, file_name):
        new_format = {}
        new_format[str(file_name)] = {
            'NLI': {
                'x': [],
                'y': []
            },
            'NMA': {
                'x': [],
                'y': []
            },
            'FLI': {
                'x': [],
                'y': []
            },
            'FMA': {
                'x': [],
                'y': []
            },
            'roi': []
        }
        return new_format

    def _parse_json(self, json_path, file_name, calc_roi=True, image_path=None):
        with open(json_path, 'r') as f:
            meta_data = json.load(f)

        new_format_near = self._get_new_dict(str(file_name))
        new_format_far  = self._get_new_dict(str(file_name))

        near_x_min, near_y_min = float('inf'), float('inf')
        near_x_max, near_y_max = 0, 0

        far_x_min, far_y_min = float('inf'), float('inf')
        far_x_max, far_y_max = 0, 0

        is_near_lima_annotated = False
        is_far_lima_annotated = False

        if image_path:
            img = cv2.imread(image_path)
        
        for line_dict in meta_data['shapes']:
            line_type = line_dict['label']

            if not (line_type in ['NLI', 'NMA', 'FLI', 'FMA']):
                raise ValueError(f'Annotated label must be NLI, NMA, FLI, or FMA, but got {line_type} in {json_path}.')

            for (x,y) in line_dict['points']:
                if line_type in ['NLI', 'NMA']: # near
                    new_format_near[str(file_name)][line_type]['x'].append(x)
                    new_format_near[str(file_name)][line_type]['y'].append(y)
                    near_x_min = min(near_x_min, x)
                    near_y_min = min(near_y_min, y)

                    near_x_max = max(near_x_max, x)
                    near_y_max = max(near_y_max, y)
                    is_near_lima_annotated = True
                    
                    
                elif line_type in ['FLI', 'FMA']: # far
                    new_format_far[str(file_name)][line_type]['x'].append(x)
                    new_format_far[str(file_name)][line_type]['y'].append(y)
                    far_x_min = min(far_x_min, x)
                    far_y_min = min(far_y_min, y)

                    far_x_max = max(far_x_max, x)
                    far_y_max = max(far_y_max, y)
                    is_far_lima_annotated = True
                    
        
                if image_path:
                    if line_type == 'NLI':
                        color = (255,0,0)
                    elif line_type == 'NMA':
                        color = (0,0,255)
                    elif line_type == 'FLI':
                        color = (0,255,0)
                    elif line_type == 'FMA':
                        color = (0,255,255)
                    img = cv2.circle(img, tuple(map(int, (x, y))), 3, color, cv2.FILLED)

                    h_loc = 50
                    for dir_name in json_path.split('/'):
                        img = cv2.putText(img, dir_name, (50, h_loc), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        h_loc += 50





        #if image_path and is_near_lima_annotated and is_far_lima_annotated:
        if image_path:
            debug_file_path = os.path.join(self.vis, image_path.split('/')[-1])
            cv2.imwrite(debug_file_path, img)
            
        
        if calc_roi: # xmin ymin xmax ymax
            if is_near_lima_annotated:
                new_format_near[str(file_name)]['roi'] = [near_x_min-self.roi_expand_thres, near_y_min-self.roi_expand_thres, near_x_max+self.roi_expand_thres, near_y_max+self.roi_expand_thres]
                new_format_near[str(file_name)]['roi'] = list(map(int, new_format_near[str(file_name)]['roi']))
            if is_far_lima_annotated:
                new_format_far[str(file_name)]['roi'] = [far_x_min-self.roi_expand_thres, far_y_min-self.roi_expand_thres, far_x_max+self.roi_expand_thres, far_y_max+self.roi_expand_thres]
                new_format_far[str(file_name)]['roi'] = list(map(int, new_format_far[str(file_name)]['roi']))

        ret = []
        if is_near_lima_annotated:
            ret.append(new_format_near)
        if is_far_lima_annotated:
            ret.append(new_format_far)

        return ret
    

    def _parse_old_json(self, old_dict, file_name):
        new_format = self._get_new_dict(file_name)

        # FLI
        for x, y in zip(old_dict['li']['x'], old_dict['li']['y']):
            new_format[file_name]['FLI']['x'].append(x)
            new_format[file_name]['FLI']['y'].append(y)
        
        # FMA
        for x, y in zip(old_dict['ma']['x'], old_dict['ma']['y']):
            new_format[file_name]['FMA']['x'].append(x)
            new_format[file_name]['FMA']['y'].append(y)

        # ROI
        for p in old_dict['roi']:
            new_format[file_name]['roi'].append(p)
        
        return new_format


    def run(self):
        sample_idx = self._get_sample_start_index()
        json_dict = {}

        if self.old_json:
            # preprocess old json file
            with open(self.old_json, 'r') as f:
                old_json_dict = json.load(f)
            for bmp_name in old_json_dict:
                json_dict.update(self._parse_old_json(old_json_dict[bmp_name], bmp_name))

        # process new annotated samples
        for i, (f_json, f_bmp) in enumerate(self._get_json_list()):
            src_file_path = f_bmp
            bmp_file_name = str(sample_idx) + '.bmp'
            dst_file_path = os.path.join(self.dst_root_dir, bmp_file_name)
            
            # copy data
            shutil.copy2(src_file_path, dst_file_path)
            # json parse
            annotations = self._parse_json(f_json, bmp_file_name, image_path=dst_file_path)
            
            if len(annotations) > 1:
                dst_file_path2 = dst_file_path.replace('.bmp', '-1.bmp')
                shutil.copy2(dst_file_path, dst_file_path2)
                        
            for i in range(len(annotations)):
                if i == 1: # 2nd dict
                    key = list(annotations[i].keys())[0]
                    new_key = key.replace('.bmp', '-1.bmp')
                    json_dict.update({
                        new_key : annotations[i][key]  
                    })
                else:
                    json_dict.update(annotations[i])

            sample_idx += 1

        # dump to json
        with open(self.dst_json, 'w') as f:
            json.dump(json_dict, f, indent='\t')

        print(f'number of samples: {len(json_dict.keys())}')

def opt():
    return EasyDict({
        'src_root_dir': './hard_case',
        'dst_root_dir': './hardsample_dataset_v4',
        'old_json': 'gTruth_pp_train.json',
        #'old_json': None,
        'dst_json': 'gTruth_pp_v4.json',
        'thres': 20,
        'vis': './parsing_results'
    })
       
def main():
    args = opt()

    app = DVC(**args)

    app.run()

if __name__ == '__main__':
    main()