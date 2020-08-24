import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split


x = np.array([[random.random()/2 for _ in range(2)] for _ in range(1000)])     #input dataset
y = np.array([i[0]+i[1] for i in x])    #target dataset

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)    #splitting dataset into training and testing parts where test-size = 30%


#Building model
perceptron_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_dim=2, activation="sigmoid"),     #Adding 1st layer connected to inputs
    tf.keras.layers.Dense(1, activation="sigmoid"),    #Adding 2nd layer(final) connected to previous layers
])

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)    #learning rule i.e. gradient descent
perceptron_model.compile(optimizer=optimizer, loss="MSE")    #Compiling model

perceptron_model.fit(x_train, y_train, epochs=500)    #Training the model

print("\nPerceptron Model testing - ")
error = perceptron_model.evaluate(x_test, y_test, verbose=1)    #Testing the model with test part

data = np.array([[0.1,0.2], [0.4, 0.3]])    #data for predictions
predict = perceptron_model.predict(data)   #predictions by model

print("\nAccording to the model, the calculation should be -")
for d,p in zip(data, predict):
    print("{} + {} = {}".format(d[0], d[1], p[0]))
