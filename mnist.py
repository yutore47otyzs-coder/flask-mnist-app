import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

import numpy as np


classes = ["0","1","2","3","4","5","6","7","8","9"]
image_size = 28

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)

# app.secret_key = "your_secret_key_here"  
# submitボタンを押した際にエラーが出た場合上の行のコメントアウトを削除し、your_secret_key_hereに任意の文字列（例:aidemy)を指定し、再度アプリケーションを実行してください。

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = load_model('./model.keras')#学習済みモデルをロード


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み、np形式に変換
            img = image.load_img(filepath, color_mode='grayscale', target_size=(image_size,image_size))
            img = image.img_to_array(img)

            # ★修正ポイント1: 正規化
            # 学習時と同じく、0〜255 を 0.0〜1.0 に変換します
            img = img / 255.0

            # ★【ここを追加！】色の反転チェック
            # 画像の平均値が高い（0.5以上）＝「全体的に白い」と判断して反転させる
            if np.mean(img) > 0.5:
                img = 1.0 - img  # 白(1.0)と黒(0.0)をひっくり返す
                print("【Info】画像を白黒反転しました", file=sys.stderr) 

                
            data = np.array([img])
            #変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            
            # ★2. print文をこれに書き換え（file=sys.stderr をつけるのがコツです）
            # ★ printの代わりに app.logger.info を使います（確実に出ます）
            #app.logger.info("========================================")
            #app.logger.info(f"File Name: {filename}")
            #app.logger.info(f"Max Value: {np.max(data)}")
            #app.logger.info(f"Result: {result}")
            #app.logger.info("========================================")

            predicted = result.argmax()
            pred_answer = "これは " + classes[predicted] + " です"

            return render_template("index.html",answer=pred_answer)

    return render_template("index.html",answer="")


if __name__ == "__main__":
   #app.run() 
   #app.run(debug=True)
   port = int(os.environ.get('PORT', 8080))
   app.run(host ='0.0.0.0',port = port)