import tensorflow as tf

def create_model(X_n,Y_n,alpha,batch_normalization = True,seperate_nets = True,normalize = False, mean = None, var = None):
    hidden_layer_nodes1 = 100
    hidden_layer_nodes2 = 100
    hidden_layer_nodes3 = 100
    hidden_layer_nodes4 = 100

    separate_layers_nodes = 20
   # tf.reset_default_graph()  
    if normalize:
        input = tf.keras.Input(shape = [X_n])
        #*tf.keras.backend.shape(input)
        mean_mat = tf.keras.backend.constant([mean])
       # mat = input - mean_mat
        mat = tf.keras.layers.Add()([input,mean_mat])
        #layer = tf.keras.layers.Lambda(
        model = tf.keras.Model(inputs=input,outputs=mat)

    input = tf.keras.Input(shape = [X_n])
    if seperate_nets:
        if batch_normalization:
            net = tf.keras.layers.BatchNormalization()(input)
            net = tf.keras.layers.Dense(hidden_layer_nodes1, activation=tf.keras.activations.relu)(input)
        else:
            net = tf.keras.layers.Dense(hidden_layer_nodes1, activation=tf.keras.activations.relu)(input)
        outputs = []
        for _ in range(Y_n):
            fc = tf.keras.layers.Dense(separate_layers_nodes, activation=tf.keras.activations.relu)(net)
            fc = tf.keras.layers.Dense(separate_layers_nodes, activation=tf.keras.activations.relu)(fc)
            outputs.append(tf.keras.layers.Dense(1)(fc))
        output = tf.keras.layers.concatenate(outputs)
        #model = tf.keras.Model(inputs=input,outputs=output_conc)
    else:
        if not batch_normalization:
            #input = tf.keras.Input(shape = [X_n])
            net = tf.keras.layers.Dense(hidden_layer_nodes1, activation=tf.keras.activations.relu)(input)
            net = tf.keras.layers.Dense(hidden_layer_nodes2, activation=tf.keras.activations.relu)(net)
            net = tf.keras.layers.Dense(hidden_layer_nodes3, activation=tf.keras.activations.relu)(net)
            output = tf.keras.layers.Dense(Y_n)(net)
           # model = tf.keras.Model(inputs=input,outputs=output)
        else:
            #input = tf.keras.Input(shape = [X_n])
            net = tf.keras.layers.BatchNormalization()(input)
            net = tf.keras.layers.Dense(hidden_layer_nodes1)(net)
            #net = tf.keras.layers.BatchNormalization()(net)
            net = tf.keras.layers.Activation('relu')(net)
            net = tf.keras.layers.Dense(hidden_layer_nodes2)(net)
            #net = tf.keras.layers.BatchNormalization()(net)
            net = tf.keras.layers.Activation('relu')(net)
            net = tf.keras.layers.Dense(hidden_layer_nodes3)(net)
            #net = tf.keras.layers.BatchNormalization()(net)
            net = tf.keras.layers.Activation('relu')(net)
            output = tf.keras.layers.Dense(Y_n)(net)

    model = tf.keras.Model(inputs=input,outputs=output)
    #model = tf.keras.models.Sequential([
    #        tf.keras.layers.Dense(hidden_layer_nodes1, activation=tf.keras.activations.relu, input_shape=(X_n,)),
    #        tf.keras.layers.Dense(hidden_layer_nodes2, activation=tf.keras.activations.relu),
    #        tf.keras.layers.Dense(hidden_layer_nodes3, activation=tf.keras.activations.relu),
    #        tf.keras.layers.Dense(Y_n)
    #        ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.mean_squared_error
            )#metrics=['mae']
    print("Network ready")

    graph =  tf.get_default_graph()#tf.compat.v1.get_default_graph()#
    return model,graph