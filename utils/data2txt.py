import numpy as np 
import os
from os.path import join
from sklearn.model_selection import train_test_split

path = '/mnt/lustre/shenrui/data/pelvis_resampled'

def arrange(path):
    '''arrange data info to txt'''
    img_ext = ['.nii', '.nii.gz']
    img_fnames = [x for x in os.listdir(path) if (x.endswith(img_ext[0]) or (x.endswith(img_ext[1])))]
    mydict = {}
    for f in img_fnames:
        fname = f.split('.')
        if 'label' in fname[0]:
            name = fname[0].split('_label')[0]
            idx = 1
        else:
            name = fname[0]
            idx = 0
        if mydict.get(name) is None:
            value = ['', '']
            value[idx] = join(path, f)
            mydict[name] = value
        else:
            mydict[name][idx] = join(path, f)
    checklist = np.where(np.array(list(mydict.values())) == '')
    if len(checklist[0]) > 0:
        for idx in checklist[0]:
            print('missing pair:', list(mydict.values())[idx])
        raise NameError('Missing File')
    return mydict

def to_str(var):
    return ('\t'.join(map(str, var))) 

def writetxt(dataset, fname):
  fw = open(fname,'w')
  for (key, value) in dataset.items():
      fw.write(key+'\t')
      #print(key)
      #print(value)
      fw.write(to_str(value)+'\n')
  fw.close()

def data_split(val_size, test_size, mydict):
  '''sperate the data into train, validation, test'''
  y = list(mydict.values())
  x = list(mydict.keys())
  x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size = test_size + val_size)
  x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size = test_size/(test_size+val_size))
  return dict(zip(x_train, y_train)), dict(zip(x_val, y_val)), dict(zip(x_test, y_test))

if __name__ == '__main__':
  mydict = arrange(path)
  trainset, valset, testset = data_split(0.15, 0.15, mydict)
  writetxt(trainset, join(path,'dataset_train.txt'))
  writetxt(valset, join(path, 'dataset_val.txt'))
  writetxt(testset, join(path, 'dataset_test.txt'))
  writetxt(mydict, join(path, 'dataset_all.txt'))
  print('train:', len(trainset))
  print('val:', len(valset))
  print('test:', len(testset))
