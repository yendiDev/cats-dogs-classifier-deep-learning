from flask import Flask, request, jsonify
from model_files.ml_model import predict_image, show_message
import numpy as np

app = Flask('cat_dog_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    res = request.get_json()
    # conver json to dictionary
    image_array = dict(res)

    # receive image array already in list format
    image_lsit = image_array.get('image')

    # convert image from list to np array
    image_ndarray = np.array(image_lsit)

    print("***********************************************\n\n")
    print("The image array is: ", image_ndarray)
    print("\n\n***********************************************")

    # receive prediction
    prediction = predict_image(image_ndarray)
    response = {
        'prediction': prediction.tolist(),
    }
    return jsonify(response)

    #return show_message('This message is returned grrr')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)