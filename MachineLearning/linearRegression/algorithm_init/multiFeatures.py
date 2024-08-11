import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lr import LinearRegression
import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode()
data = pd.read_csv('../data/world-happiness-report-2017.csv')
#拆分训练集和测试集
train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index)

input_param_name1 = 'Economy..GDP.per.Capita.'
input_param_name2 = 'Freedom'

output_param_name = 'Happiness.Score'

x_train = train_data[[input_param_name1],[input_param_name2]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name1],[input_param_name2]].values
y_test = test_data[[output_param_name]].values

plot_training_trace = go.Scatter3d(
    x=x_train[:, 0].flatten(),
    y=x_train[:,1].flatten(),
    z=y_train.flatten(),
    name = 'Training Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line':{
            'color':'rgb(255,255,255)',
            'width':1
        }
    }
)
plot_test_trace = go.Scatter3d(
    x=x_test[:, 0].flatten(),
    y=x_test[:,1].flatten(),
    z=y_test.flatten(),
    name = 'Test Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line':{
            'color':'rgb(255,255,255)',
            'width':1
        }
    }
)

plot_layout = go.Layout(
    title='Data Sets',
    scene={

    }
)
num_iterations = 500
learning_rate = 0.01
linear_regression = LinearRegression(x_train,y_train)
(theta,cost_history) = linear_regression.train(learning_rate, num_iterations)

print('开始时损失：',cost_history[0])
print('训练后损失：',cost_history[-1])

plt.plot(range(num_iterations),cost_history)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.title("GD")
plt.show()

predictions_num = 100
x_prediction = np.linspace(min(x_train),max(x_train),predictions_num).reshape(predictions_num,1)
y_prediction = linear_regression.predict(x_prediction)

plt.scatter(x_train, y_train, label  = 'Train_data')
plt.scatter(x_test, y_test, label = 'test_data')
plt.plot(x_prediction,y_prediction,'r',label = 'Prediction')

plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title("visual")
plt.legend()
plt.show()