import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import glob
selectsignlecsv=1
from sklearn.model_selection import train_test_split
DOIMGPROCESS=1
def MakeAllSubdir(DATA_PATH,parent_path=None):
    if parent_path==None:
        parent_path = os.path.dirname(os.getcwd())
    DATA_PATH=DATA_PATH.split(parent_path)[1]
    if not os.access(DATA_PATH, os.W_OK):
        folders2add = DATA_PATH.split('/')
        last = ''
        for each in folders2add:
            if len(each):
                if (not os.access(last + each, os.W_OK)):
                    os.mkdir(last + each)
                if len(last)==0:
                    last=each+'/'
                else:
                    last = last + each + '/'


def rename_file(f, s):
    # 2017.jpg -> 2017_front.jpg
    return f.split('.')[0] + s + '.' + f.split('.')[1]

parent_path = os.path.dirname(os.getcwd())
csv_input_dir_path = os.path.join(parent_path,'scripts', 'data', 'csv')
csv_preprocess_dir_path = os.path.join(csv_input_dir_path, 'preprocess')
csv_output_dir_path = os.path.join(csv_input_dir_path, 'final')
#prepare all subdirs:
MakeAllSubdir(csv_input_dir_path,parent_path=parent_path+'/scripts')
MakeAllSubdir(csv_preprocess_dir_path,parent_path=parent_path+'/scripts')
MakeAllSubdir(csv_output_dir_path,parent_path=parent_path+'/scripts')
AllCSVFiles=glob.glob(csv_input_dir_path+'/*.csv')
if selectsignlecsv:
    AllCSVFiles=AllCSVFiles[3:4]
if DOIMGPROCESS:
    for CurrentCsvFile in AllCSVFiles:
        file_name = CurrentCsvFile.split('.csv')[0]
        file_name =file_name.split('/')[-1]
        #prepare folders:
        csv_file_name = file_name + '.csv'
        csv_output_file_name = file_name + '_preprocess.csv'
        csv_file_path = os.path.join(csv_input_dir_path, csv_file_name)
        csv_output_file_path = os.path.join(csv_preprocess_dir_path, csv_output_file_name)
        img_base_path = os.path.join(parent_path,'scripts', 'data', 'img')
        img_dir_path = os.path.join(img_base_path, 'raw')
        img_front_dir_path = os.path.join(img_base_path, 'front')
        img_left_dir_path = os.path.join(img_base_path, 'side_left')
        img_right_dir_path = os.path.join(img_base_path, 'side_right')
        MakeAllSubdir(img_front_dir_path,parent_path=parent_path+'/scripts')
        MakeAllSubdir(img_left_dir_path,parent_path=parent_path+'/scripts')
        MakeAllSubdir(img_right_dir_path,parent_path=parent_path+'/scripts')

        df = pd.read_csv(csv_file_path, header=0)
        print("%d rows" % df.shape[0])
        df.head(3)
        df.tail(3)
        img_path = os.path.join(img_dir_path, df['img'][0])
        img = Image.open(img_path)
        print("image size is %s" % (img.size,))
        fig = plt.figure(figsize = (10,10))
        plt.imshow(img)
        plt.show()
        # define coordinates
        front_coord = (345,217, 951, 517)#(345,217, 951, 517)
        left_coord = (17, 136, 194, 374)
        right_coord = (955, 136, 1132, 374)
        img_front = img.crop(front_coord)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set(title='front view')
        ax.imshow(img_front)
        img_left = img.crop(left_coord)
        img_right = img.crop(right_coord)
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(7,7))
        ax0.set(title='left side mirror view')
        ax0.imshow(img_left)
        ax1.set(title='right side mirror view')
        ax1.imshow(img_right)
        plt.show()
        img_front_path_list = []
        img_left_path_list = []
        img_right_path_list = []
        for img_filename in df['img']:
            img_path = os.path.join(img_dir_path, img_filename)
            img = Image.open(img_path)

            img_front = img.crop(front_coord)
            img_left = img.crop(left_coord)
            img_right = img.crop(right_coord)

            img_front_path = rename_file(img_filename, '_front')
            img_left_path = rename_file(img_filename, '_left')
            img_right_path = rename_file(img_filename, '_right')

            img_front_path_list.append(img_front_path)
            img_left_path_list.append(img_left_path)
            img_right_path_list.append(img_right_path)

            img_front.save(os.path.join(img_front_dir_path, img_front_path))
            img_left.save(os.path.join(img_left_dir_path, img_left_path))
            img_right.save(os.path.join(img_right_dir_path, img_right_path))
        df['front'] = img_front_path_list
        df['side_left'] = img_left_path_list
        df['side_right'] = img_right_path_list
        df.head(3)
        df.to_csv(csv_output_file_path, index=False)
        df = pd.read_csv(csv_output_file_path, header=0)
        print("%d rows" % df.shape[0])
        df.head(3)
    #os.rename(csv_file_path, os.path.join(csv_preprocess_dir_path, csv_file_name))



#make final csv's:
AllCSVFiles=glob.glob(csv_preprocess_dir_path+"/*.csv")
if selectsignlecsv:
    AllCSVFiles=AllCSVFiles[3:4]
i=0
for each in AllCSVFiles:
    if i==0:
        df_final=pd.read_csv(each)
        i=1
    else:
        df=pd.read_csv(each)
        df_final=pd.concat([df_final ,df],ignore_index=True)


cur_final_file = os.path.join(csv_output_dir_path, 'e1.csv')

df_final.to_csv(cur_final_file, mode='w', index=False, header=True)
print("end")
df_train, df_valid = train_test_split(df_final, test_size = 0.2)
df_train.to_csv(os.path.join(csv_output_dir_path, 'v1_train.csv'), index=False)
df_valid.to_csv(os.path.join(csv_output_dir_path, 'v1_valid.csv'), index=False)
