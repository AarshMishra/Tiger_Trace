# Tiger_Trace

---

# Tiger Conservation and Monitoring

The International Union for Conservation of Nature (IUCN) classifies tigers as 'endangered.' With over half of the world's tiger population residing in India, effective monitoring and accurate counting of these animals are crucial for their conservation. In India, the National Tiger Conservation Authority (NTCA) and the Wildlife Institute of India (WII) are responsible for this vital task. The current approach involves deploying trap cameras equipped with motion detectors in national parks and tiger reserves. These cameras automatically capture images of tigers as they move through their natural habitat. Biologists and scientists then meticulously analyze these photographs to identify and distinguish individual tigers based on their unique stripe patterns. This method, while reducing direct human-wildlife interaction, is both time-consuming and labor-intensive, often leading to delays in reporting and potential inaccuracies in the tiger population data.

---
![image](https://github.com/user-attachments/assets/cdaf8bf1-92b8-4c45-8529-4c1e209a729f)
---
This deep learning model aims to address the challenges by automating the identification of tigers based on their stripe patterns. By utilizing convolutional neural networks (CNNs), it can distinguish between individual tigers with high accuracy. This innovative approach not only streamlines the monitoring process but also reduces human error, providing more reliable data for conservation efforts.

Benefits:
High accuracy in distinguishing individual tigers, streamlines the monitoring process, reduces human error, and provides reliable data for conservation efforts.


## Project Overview

Tiger_Trace is an AI-based solution aimed at automating the identification and monitoring of tigers using deep learning techniques. Our solution leverages image recognition models to detect and identify individual tigers from camera trap images, thereby reducing the manual effort involved and increasing the accuracy and speed of monitoring.

### Features
- **Automated Tiger Detection:** Using convolutional neural networks (CNNs) to detect tigers in images.
- **Individual Identification:** Identifies individual tigers based on their unique stripe patterns.
- **Real-Time Processing:** Enables quicker data processing for immediate analysis and decision-making.

## Data Preprocessing

1. **Loading and Cleaning Data:**
   - The raw image data is loaded using Python scripts (`Loading and preprocessing.py`).
   - The images are resized and normalized to ensure uniformity across the dataset.
   - Data augmentation techniques such as rotation, flipping, and cropping are applied to increase the diversity of the training data.
   
2. **Splitting the Dataset:**
   - The data is split into training and test sets using a predefined ratio (e.g., 80:20).
   - The script handles this using the `reid_list_train.csv` and `reid_list_test.csv` files, which contain metadata about the images.

3. **Feature Extraction:**
   - Important features like edges, textures, and shapes are extracted from the images to help the model learn more effectively.
   - These features are used to create a feature map that serves as the input for the model.

## Model Training

1. **Model Architecture:**
   - The model (`Model training.py`) is built using deep learning frameworks such as TensorFlow or PyTorch.
   - It uses a convolutional neural network (CNN) architecture tailored for image classification and object detection tasks.
   - The network consists of several convolutional layers, followed by max-pooling layers, fully connected layers, and a final softmax layer for classification.

2. **Training Process:**
   - The model is trained using the training dataset, and hyperparameters like learning rate, batch size, and number of epochs are configured.
   - The training process involves backpropagation and optimization techniques like Adam or SGD to minimize the loss function.
   - Early stopping and model checkpointing are implemented to prevent overfitting and ensure the best model is saved.

3. **Validation and Testing:**
   - The model's performance is validated using the test set. Metrics such as accuracy, precision, recall, and F1-score are calculated.
   - Fine-tuning is performed by adjusting hyperparameters and retraining the model, if necessary.

## Testing and Predictions

1. **Testing the Model:**
   - The script (`Testing and Predictions.py`) loads the trained model and runs it against the test set.
   - Predictions are made, and results are saved in `submit.csv`.

2. **Generating Predictions:**
   - The model's predictions are analyzed to identify individual tigers in new images.
   - The results are displayed with bounding boxes and confidence scores.

## Pre Trained Model
Here are the pretrained models on the given dataset:

Resnet: [here](https://1drv.ms/f/s!AjMzlbqC81_ZapGumRTu-EsxWkI?e=VqcIDM)


EfficientNb3:[here](https://1drv.ms/f/s!AjMzlbqC81_ZapGumRTu-EsxWkI?e=VqcIDM)


## Usage

To run this project, you need to have Python 3. x installed, along with the necessary libraries (TensorFlow, PyTorch, OpenCV, etc.). Follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/Tiger_Trace.git
   cd Tiger_Trace
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the data preprocessing script:
   ```bash
   python "Loading and preprocessing.py"
   ```

4. Train the model:
   ```bash
   python "Model training.py"
   ```

5. Test the model and generate predictions:
   ```bash
   python "Testing and Predictions.py"
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to Dr. Wee Kek Tan for mentorship and guidance during the project.
- Sponsored by the government of Chhattisgarh and the Biodiversity Board.
- Thanks to the National University of Singapore for the research internship opportunity.

---





