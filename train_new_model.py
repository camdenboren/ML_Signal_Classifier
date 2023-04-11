from pymongo import MongoClient
import json
from matplotlib import pyplot as plt
import tensorflow as tf

def train_new_model():
    user_choice = input("1. Select data by modulation type\n2. Go to main menu\n")
    try:
        user_choice = int(user_choice)
    except:
        pass
    else:
        if(user_choice == 1):
            trainModel()
        elif(user_choice == 2):
            return
        else:
            print("Invald option...")

def trainModel():
    inputMods = [int(item) for item in input("Please Enter a List of input modulations for training 1 3 5 \
                        \n1. OOK\n2. 4ASK\n3. 8ASK\n4. BPSK \
                        \n5. QPSK \n6. 8PSK\n7. 16PSK \
                        \n8. 32PSK \n9. 16APSK \
                        \n10. 32APSK\n11. 64APSK\n12. 128APSK\n13. 16QAM\n14. 32QAM \
                        \n15. 64QAM\n16. 128QAM\n17. 256QAM\n18. AM-SSB-WC\n19. AM-SSB-SC \
                        \n20. AM-DSB-WC\n21. AM-DSB-SC\n22. FM\n23. GMSK\n24. OQPSK\n").split()]                #OOK 4ASK BPSK QPSK 16QAM
    print("Collecting Data might take some time.\n")
    modList, valList, totalLength = pullModulationList(inputMods)
    train_ds = datasetManip(modList, 800, drop_rem=True)
    val_ds = datasetManip(valList, 100, drop_rem=True)
    _name = input("Enter name of New Model: ")
    model = trainNewModel(train_ds, val_ds, outNode=totalLength, name=_name)
    createJsonFile(_name, inputMods)

def datasetManip(dset, batch_s=500, buf_s=100000, drop_rem=True):
    dset_trans = dset.shuffle(buffer_size=buf_s).batch(batch_s, drop_remainder=drop_rem)
    return dset_trans

def pullModulationList(modulations, sampleSize=9000, valSize=1000):
    # Pulling data for train and Validation. 
    # Pulling 1000 for train and 100 for Validation
    # For comparison nate-model uses 340787 data entries to train. 

    client = MongoClient('143.244.155.10', 8080)
    db = client.test #this is the database name
    # _collection = db['test_store_signals'] #this is the collection name
    _collection = db['test_training_signals']

    #grab total length of the list
    totalLength = len(modulations)
    #going to extend data to this list to create the whole list
    #need to randomize after gathering all the data
    completeModulationList = []
    trainIQ = []
    trainMod = []
    completeValidationList = []
    valIQ = []
    valMod = []

    #pull [sampleSize] for each [modulation] and give the hotEncoded data
    for i, mod in enumerate(modulations):
        currentMod = (_collection.aggregate([
            #Not sure what the key will be named at the moment
            {'$match': {'intOf_Modulation': mod}},
            {'$sample': {'size': sampleSize}}
        ], allowDiskUse=True))
        for item in currentMod:
            item['modulation'] = hotEncodedList(totalLength, mod, i)
            #At the currennt moment I am assuming the [item] is a list of lists
            #So I am appennding a hot encoded list to the end of said [item]
            trainIQ.append(item['iq-frame'])
            trainMod.append(item['modulation'])
        #add new list to the end of [completeModulationList]

    for i, mod in enumerate(modulations):
        currentMod = (_collection.aggregate([
            #Not sure what the key will be named at the moment
            {'$match': {'intOf_Modulation': mod}},
            {'$sample': {'size': valSize}}
        ]))
        for item in currentMod:
            item['modulation'] = hotEncodedList(totalLength, mod, i)
            #At the currennt moment I am assuming the [item] is a list of lists
            #So I am appennding a hot encoded list to the end of said [item]
            valIQ.append(item['iq-frame'])
            valMod.append(item['modulation'])
        #add new list to the end of [completeModulationList]
        
    completeModulationList = tf.data.Dataset.from_tensor_slices((trainIQ, trainMod))
    completeValidationList = tf.data.Dataset.from_tensor_slices((valIQ, valMod))

    #Need to check what the list look like so I can understand how it's being pulled
    #Shuffle the [completeModulationList] for training purposes
    client.close()
    return (completeModulationList, completeValidationList, totalLength)

def totalHotEncoded(modulations):
    #assuming the last element is the highest number
    #if this is not the case we can create a function to sort said list or use predefined method if python has that
    totalLength = 24
    
    if(modulations[len(modulations)-1] > 24):
        totalLength = modulations[len(modulations)-1]
        return totalLength
    else:
        return totalLength


#Create array for the modulation with hot encoded data
def hotEncodedList(totalLength, hotSpot, index=-1):

    if (index < 0):
        arrayList = [0] * totalLength
        arrayList[hotSpot-1] = 1
        return arrayList
    else:
        arrayList = [0] * totalLength
        arrayList[index] = 1
        return arrayList

def trainNewModel(trn, val, outNode=24, name='newModel', optimizer='Adam', loss_function='categorical_crossentropy', num_epochs=15):
    new_modulation_model = tf.keras.models.Sequential([
      tf.keras.layers.Conv1D(64, 3, activation='relu', 
                  input_shape=(1024, 2)),
      tf.keras.layers.Conv1D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling1D(2),
      tf.keras.layers.Conv1D(64, 3, activation='relu'),
      tf.keras.layers.Conv1D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling1D(2),
      tf.keras.layers.Conv1D(64, 3, activation='relu'),
      tf.keras.layers.Conv1D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling1D(2),
      tf.keras.layers.Conv1D(64, 3, activation='relu'),
      tf.keras.layers.Conv1D(64, 3, activation='relu'),
      tf.keras.layers.MaxPooling1D(2),
      tf.keras.layers.Conv1D(64, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(128, activation=tf.nn.selu),
      tf.keras.layers.Dense(128, activation=tf.nn.selu),
      tf.keras.layers.Dense(outNode, activation=tf.nn.softmax)
    ])

    new_modulation_model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    new_modulation_model.summary()

    model_history = new_modulation_model.fit(x=trn, epochs=num_epochs, validation_data=val, verbose=1)
    
    plotHistory(model_history, name)

    new_modulation_model.save(f'.\\models\\{name}')

    return new_modulation_model

    
def plotHistory(hist, plot_title):
    #summarize history for accuracy
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    
    full_title = plot_title + " accuracy"
    plt.title(full_title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    plt.show()
    #summarize history for loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    
    full_title = plot_title + " loss"
    plt.title(full_title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    plt.show()

def createJsonFile(name, modlist):
    modulation_classes = json.load(open("./classes-fixed.json", 'r')) # list of modulation classes
    list = []

    for x in modlist:
        list.append(modulation_classes[int(x)-1])

    json_obj = json.dumps(list, indent=4)

    with open(".\\models\\"+str(name)+"\\"+str(name)+".json", "w") as outfile:
        outfile.write(json_obj)


# def select_by_modulation():
#      while(True):
#         try:
#             user_mod = int(input("Please choose a modulation from this list \
#                         \n1. OOK\n2. 4ASK\n3. 8ASK\n4. BPSK \
#                         \n5. QPSK \n6. 8PSK\n7. 16PSK \
#                         \n8. 32PSK \n9. 16APSK \
#                         \n9. 32APSK\n10. 64APSK\n11. 128APSK\n12. 16QAM\n13. 32QAM \
#                         \n14. 64QAM\n15. 128QAM\n16. 256QAM\n17. AM-SSB-WC\n18. AM-SSB-SC \
#                         \n19. AM-DSB-WC\n20. AM-DSB-SC\n21. FM\n22. GMSK\n23. OQPSK\n"))
#             user_sample = int(input("Please enter the number of sample modulations you would like: \n"))
#         except:
#             pass
#         else:
#             if(user_mod >= 1 or user_mod <= 23):
#                 client = MongoClient('143.244.155.10', 8080)
#                 db = client.test #this is the database name
#                 _collection = db['test_store_signals']
#                 mod_array = get_mod_arr(user_mod)

#                 modulationQueriedList = list(_collection.aggregate([
#                     { '$match': {'modulation': mod_array} },
#                     { '$sample': { 'size': user_sample } }
#                     ]))
                
#                 # send to train new model ?
                
#                 break
#             else:
#                 print("User input is incorrect")

# def get_mod_arr(i): 
#     arr = [0] * 24
#     arr[i-1] = 1
#     return arr

# modulation_classes = json.load(open("./classes-fixed.json", 'r')) # list of modulation classes

# def get_mod_from_arr(x):
#     return modulation_classes[x.index(1)];
