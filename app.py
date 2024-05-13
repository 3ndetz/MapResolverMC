import gradio as gr
import io
def solve_captcha(data_bytes: bytes = None, file_name: str = None):
    import time
    import os
    import sys
    import cv2
    import numpy as np
    import onnxruntime


    real_tests = False
    from config import characters, img_height, img_width, img_type, max_length, transpose_perm, OUTPUT_ONNX

    sess = onnxruntime.InferenceSession(f"out.model.onnx")
    name = sess.get_inputs()[0].name
    def get_result(pred):
        """CTC decoder of the output tensor
        https://distill.pub/2017/ctc/
        https://en.wikipedia.org/wiki/Connectionist_temporal_classification
        :return string, float
        """
        accuracy = 1
        last = None
        ans = []
        # pred - 3d tensor, we need 2d array - first element
        for item in pred[0]:
            # get index of element with max accuracy
            char_ind = item.argmax()
            # ignore duplicates and special characters
            if char_ind != last and char_ind != 0 and char_ind != len(characters)+1:
                # this element is a character - append it to answer
                ans.append(characters[char_ind - 1])
                # Get accuracy for current character and multiply global accuracy by it
                accuracy *= item[char_ind]
            last = char_ind

        answ = "".join(ans)[:max_length]
        return answ, accuracy


    def decode_img(data_bytes: bytes):
        # same actions, as for tensorflow
        image = cv2.imdecode(np.asarray(bytearray(data_bytes), dtype=np.uint8), 1)
        image: "np.ndarray" = image.astype(np.float32) / 255.
        if image.shape != (img_height, img_width, 3):
            image = cv2.resize(image, (img_width, img_height))
        image = image.transpose(transpose_perm)
        #  Creating tensor ( adding 4d dimension )
        image = np.array([image])
        return image
        
    def decode_img_array(nump_array):
        # same actions, as for tensorflow
        #image = cv2.imdecode(nump_array, 1)
        image: "np.ndarray" = nump_array.astype(np.float32) / 255.
        if image.shape != (img_height, img_width, 3):
            image = cv2.resize(image, (img_width, img_height))
        image = image.transpose(transpose_perm)
        #  Creating tensor ( adding 4d dimension )
        image = np.array([image])
        return image

    def solve(data_bytes: bytes=None, file_name=None):
        if file_name:
            with open(file_name, 'rb') as F:
                data_bytes = F.read()
        if data_bytes is None:
            print('[CAPTCHA RESOLVER NN] ПУСТОТА ВМЕСТО БАЙТОВ!')
            return None
        if isinstance(data_bytes,np.ndarray):
            print('Img is ndarray')
            img = decode_img_array(data_bytes)
        else:
            print('Img is bytes')
            img_byte_arr = io.BytesIO()
            data_bytes.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            img = decode_img(data_bytes)
        #print(img)
        pred_onx = sess.run(None, {name: img})[0]
        ans = get_result(pred_onx)
        return ans
    result = solve(data_bytes,file_name)
    print ('solved', result)
    return {"result":result[0], "predict":result[1]}
    
def image_classifier(inp):
    if inp is None:
        return "Не загружена картинка"
    print('INPUT GOT>>>', inp)
    #gradio img inp - numpy ndarray
    
    
    result = solve_captcha(inp)
    return result
#inputs=gr.Image(type="pil")
examples_names = ["1089","3569","5989","18769","73295","89238"]
examples = []
for n in examples_names:
    examples.append("examples/"+n+".png")
demo = gr.Interface(fn=image_classifier, inputs=gr.Image(), outputs="textbox", examples=examples)
demo.launch(show_api = True)