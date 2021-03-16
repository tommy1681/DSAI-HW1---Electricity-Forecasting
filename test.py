import tensorflow as tf 
import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from data_obj import data_obj
#units,表示每个时间步的输出维度，比如输入的每个时间步是一个3维向量，则输出是一个6维向量
#return_sequences，控制是否输出全部时间步的隐藏层（h_t）
#return_state，控制是否输出最后一个时间步的状态(c_t)
#stateful,控制记忆状态是否在不同样本间保持，默认False,不保持
#recurrent_activation:门函数的激活函数（sigmoid/hard_softmax）
#activation：输出函数(隐藏状态)的激活函数/内部状态函数的激活函数,常用tanh,比如隐藏层输出 h_t = Z_o * tanh * c_t，内部函数的激活函数也是用tanh
class Model:
    def __init__(self):
        a=1
    def build(self,data):
        
        model = tf.keras.Sequential([
        
        tf.keras.layers.LSTM(500,input_shape=data.shape[-2:]),
        tf.keras.layers.Dense(7)])

        
        model.compile(optimizer='sgd', loss='mean_squared_error')
        return model

    def train(self,data):
        
        self.data = data
        print("資料前處理")
        x,y= self.data_pre_new(data)
        x,y =self.unison_shuffled_copies(x,y)
        x,y,test_x,test_y = self.cut(x,y)
        model =self.build(x)
        model.fit(x,y,validation_data = [test_x,test_y] ,epochs=400)
        model.save('path_to_saved_model', save_format='tf')
        print("模型儲存完畢")


    def data_pre_new (self,data):
        h = data.shape[0]
        data = data.fillna(value=0)
        target = []
        for i in range(0,h):
            target.append(data["備轉容量(MW)"][i]/2000)
            
        x = []
        y = []
        count = 0
        
        for key in data:
        
            uni_data = data[key]
            uni_data = uni_data.values
            uni_train_max = uni_data.max()
            uni_train_min = uni_data.min()
            uni_train_min_max = uni_train_max - uni_train_min
            
            if uni_train_min_max!= 0:
                data[key] = (uni_data-uni_train_min)/uni_train_min_max

        
        for i in range(0,h):
            if i + 29 > 762-1 :
                break
            batch = []
            #前21天
            for j in range(0,21):

                ans = []
                for key in data:
                    if key != "備轉容量(MW)":
                        a = data[key][i+j]
                        ans.append(a)

                batch.append(ans)
            x.append(batch)
            ans = []
            #第70到76天
            for k in range(22,29):
                ans.append(target[i+k])
            y.append(ans)
            x_n = np.array(x)
            y_n = np.array(y)
        

        return x_n,y_n



    def predict_data_pre(self,data):
        h = data.shape[0]
        data = data.fillna(value=0)
        
        for key in data:
        
            uni_data = data[key]
            uni_data = uni_data.values
            uni_train_max = uni_data.max()
            uni_train_min = uni_data.min()
            uni_train_min_max = uni_train_max - uni_train_min
            if uni_train_min_max!= 0:
                data[key] = (uni_data-uni_train_min)/uni_train_min_max
        batch = []
        for j in range(h-21,h):

                ans = []
                for key in data:
                    if key != "備轉容量(MW)":
                        a = data[key][j]
                        ans.append(a)

                batch.append(ans)
        x = [batch]
        return np.array(x) 

    def cut(self,x_n,y_n):
        h = x_n.shape[0]
        spli = int(0.9*h)
        tran_x = x_n[:spli]
        test_x =  x_n[spli:]
        tran_y = y_n[:spli]
        test_y =  y_n[spli:]

        return tran_x,tran_y,test_x,test_y


    def unison_shuffled_copies(self,a, b):
        
        np.random.seed(0)
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def predict(self,n_step,data):
        x = self.predict_data_pre(data)
        new_model = tf.keras.models.load_model('path_to_saved_model')
        predict = new_model.predict(x)
        predict *=2000
        predict = predict.astype('int32')
        return data_obj(predict)
        

'''
def rmse(predictions, targets): 
    return np.sqrt(((predictions - targets) ** 2).mean()) 


df = pd.read_csv("HW1_5.csv",encoding='utf-8')

m = Model()
m.train(df)


new_model = tf.keras.models.load_model('path_to_saved_model')

x,y= m.data_pre_new(df)
x,y =m.unison_shuffled_copies(x,y)
x,y,test_x,test_y = m.cut(x,y)

predict = new_model.predict(test_x)


predict *= 2000
test_y *= 2000
print(f"均方误差(MSE)：{mean_squared_error(predict,test_y)}")
print(f"根均方误差(RMSE)：{np.sqrt(mean_squared_error(test_y,predict))}")
predict = predict.astype('int32')
print(predict)
print(test_y)

predict = m.predict(7,df)
print(predict)



#data = np.random.rand(10,3,3)

#print("---",data)




df = pd.read_csv("data.csv")


UNITS = 10

data = np.random.rand(10,3,3)
data_y = np.random.rand(10,5)
print("---",data.shape[-2:])
model = tf.keras.Sequential([
    #只输出最后一个时间步的隐藏层，是一个UNITS维度的向量，输出shape(1,6)
    tf.keras.layers.LSTM(UNITS,input_shape=data.shape[-2:]),
    tf.keras.layers.Dense(5)
])
model.compile(optimizer='adam', loss='mean_squared_error')

EVALUATION_INTERVAL = 200
EPOCHS = 10



model.fit(data,data_y, epochs=EPOCHS)

# 輸出 SavedModel 的格式來輸出模型
#model.save('path_to_saved_model', save_format='tf')

# 載入模型並建立模型物件（不需要模型的類別宣告原始碼，即可完成）
new_model = tf.keras.models.load_model('path_to_saved_model')


predict = new_model.predict(data)
print(predict)
print(predict.shape)

'''