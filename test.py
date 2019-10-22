from mlp import *


def plot_data(X, y_predict):
        
    fig, ax = plt.subplots(figsize=(12,8))
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

    indices_0 = [k for k in range(0, X.shape[0]) if y_predict[k] == 0]
    indices_1 = [k for k in range(0, X.shape[0]) if y_predict[k] == 1]
    indices_2 = [k for k in range(0, X.shape[0]) if y_predict[k] == 2]

    ax.plot(X[indices_0, 0], X[indices_0,1], marker='o', linestyle='', ms=5, label='0')
    ax.plot(X[indices_1, 0], X[indices_1,1], marker='o', linestyle='', ms=5, label='1')
    ax.plot(X[indices_2, 0], X[indices_2,1], marker='o', linestyle='', ms=5, label='2')

    ax.legend()
    ax.legend(loc=2)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Tricky 3 Class Classification')
    plt.show()



def main():
    # Let's make up our 2D data for our three classes.
    data = pd.DataFrame(np.zeros((5000, 3)), columns=['x1', 'x2', 'y'])

    # Let's make up some noisy XOR data to use to build our binary classifier
    for i in range(len(data.index)):
        x1 = random.randint(0,1)
        x2 = random.randint(0,1)
        if x1 == 1 and x2 == 0:
            y = 0
        elif x1 == 0 and x2 == 1:
            y = 0
        elif x1 == 0 and x2 == 0:
            y = 1
        else:
            y = 2
        x1 = 1.0 * x1 + 0.20 * np.random.normal()
        x2 = 1.0 * x2 + 0.20 * np.random.normal()
        data.iloc[i,0] = x1
        data.iloc[i,1] = x2
        data.iloc[i,2] = y
        
    for i in range(int(0.25 *len(data.index))):
        k = np.random.randint(len(data.index)-1)  
        data.iloc[k,0] = 1.5 + 0.20 * np.random.normal()
        data.iloc[k,1] = 1.5 + 0.20 * np.random.normal()
        data.iloc[k,2] = 1

    for i in range(int(0.25 *len(data.index))):
        k = np.random.randint(len(data.index)-1)  
        data.iloc[k,0] = 0.5 + 0.20 * np.random.normal()
        data.iloc[k,1] = -0.75 + 0.20 * np.random.normal()
        data.iloc[k,2] = 2
        
    # Now let's normalize this data.
    data.iloc[:,0] = (data.iloc[:,0] - data['x1'].mean()) / data['x1'].std()
    data.iloc[:,1] = (data.iloc[:,1] - data['x2'].mean()) / data['x2'].std()      
    data.head()

    # set X (training data) and y (target variable)
    cols = data.shape[1]
    X = data.iloc[:,0:cols-1]
    y = data.iloc[:,cols-1:cols]

    X = np.matrix(X.values)
    y = np.matrix(y.values)


    # Let's make a sloppy plotting function for our binary data.
    # plot_data(X, y)

    x = np.asarray(X)
    Y = np.asarray(y)
    label = [[1,0,0],[0,1,0],[0,0,1]]
    Y = np.array([label[int(i)] for i in list(Y.T[0])])

    # Let's start trainning!
    NN = MLP()
    NN.AddLayer('Hidden', input_dim=2, output_dim=3, act_fun = 'Relu')
    NN.AddLayer('Hidden', input_dim=3, output_dim=8, act_fun = 'Relu')
    NN.AddLayer('Hidden', input_dim=8, output_dim=16, act_fun = 'Relu')
    NN.AddLayer('Output', input_dim=16, output_dim=3, act_fun = None)
    NN.AddLayer('Loss', input_dim=3, output_dim=3, act_fun = None)

    NN.Train(x, Y, num_epoch=300, batch_size=200, lr=0.001, alpha=0.0)
    DrawMap(NN,X,y)





if __name__ == "__main__":
    main()