import tensorflow as tf

def create_model(X_n,Y_n,alpha):
    hidden_layer_nodes1 = 100
    hidden_layer_nodes2 = 100
    hidden_layer_nodes3 = 100
    hidden_layer_nodes4 = 100
    model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(hidden_layer_nodes1, activation=tf.keras.activations.relu, input_shape=(X_n,)),
            tf.keras.layers.Dense(hidden_layer_nodes2, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(hidden_layer_nodes3, activation=tf.keras.activations.relu),
            tf.keras.layers.Dense(Y_n)
            ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.mean_squared_error
            )#metrics=['mae']
    print("Network ready")
    graph = tf.get_default_graph()
    return model,graph