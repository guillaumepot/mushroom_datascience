


def gen_dataset(set, training_data, img_dim, augment, normalize, batch_size):
    
    # Encodage de la variable 'label'
    s = LabelEncoder()
    target = s.fit_transform(training_data.label)
    print('target encodée:', target)




    X_train, X_temp, y_train, y_temp = train_test_split(training_data.image_url, target, test_size = 0.3, random_state = 10)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.2, random_state = 10)



    if set == 'train':
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.map(lambda x, y : [load_image(x, resize = img_dim, augment = augment, normalize = normalize), y], num_parallel_calls=-1)
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size = dataset.cardinality())        
        dataset = dataset.batch(batch_size)

        return dataset








    elif set == 'validation':
        dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        dataset = dataset.map(lambda x, y : [load_image(x, resize = img_dim, augment = augment, normalize = normalize), y], num_parallel_calls=-1)
        dataset = dataset.batch(batch_size)
        return dataset
    

    
    elif set == 'test':
        dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        dataset = dataset.map(lambda x, y : [load_image(x, resize = img_dim, augment = augment, normalize = normalize), y], num_parallel_calls=-1)
        dataset = dataset.batch(batch_size)
        return dataset

    else:
        return print("L'argument set= n'est pas valide. Réitérer avec 'train', 'test' ou 'validation'")