import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

def main():
    # data preparation


    # CNN model
    model = Sequential()
    model.add(Conv2D(32, (5,5), activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    history = model.fit(X_train, y_train, batch_size=100, epochs=5, verbose=1, validation_split= 0.2)
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)



if __name__ == '__main__':
    main()