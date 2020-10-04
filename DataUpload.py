from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import xlrd 

loc = ("add file location") 

workbook = xlrd.open_workbook(loc) 
sheet = workbook.sheet_by_index(0) 
sheet.cell_value(0, 0)  
print(sheet.ncols) 
