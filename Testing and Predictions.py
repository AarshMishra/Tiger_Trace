test_loss, test_acc = model.evaluate(test_gen)
print('Test accuracy:', test_acc)

from tensorflow.keras.preprocessing import image
img = image.load_img(r"C:\Users\thars\OneDrive\Desktop\project\data\test\000000.jpg",target_size=(200,250))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
from tensorflow.keras.applications.resnet50 import preprocess_input
x=preprocess_input(x)

predictions = model.predict(x)

odel = load_model(model_path)

# Generate predictions
preds = model.predict(test_gen, steps=test_steps, verbose=1)

# Map predictions to class labels
classes = list(train_gen.class_indices.keys())  # Ensure this matches your training generator's class indices
pred_classlist = []
file_list = []

for i, p in enumerate(preds):
    index = np.argmax(p)
    klass = classes[index]
    pred_classlist.append(klass)
    file = test_gen.filenames[i]
    filename = os.path.basename(file)
    file_list.append(filename)

# Create a DataFrame for the submission
Fseries = pd.Series(file_list, name='file')
Lseries = pd.Series(pred_classlist, name='Class')
submit_df = pd.concat([Fseries, Lseries], axis=1)

# Save the submission DataFrame to a CSV file
submit_path = os.path.join(working_dir, 'submit.csv')
submit_df.to_csv(submit_path, index=False)

# Read the CSV file back in to make sure it is correct
submit_df = pd.read_csv(submit_path)
print(submit_df.head())
