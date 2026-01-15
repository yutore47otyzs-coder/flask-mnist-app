import os
import sys # ★追加：これがないとエラーになります
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

# ★修正：コメントアウトを外して有効化しました
app.secret_key = "super_secret_key_mnist" 

# ★追加：Render上でフォルダがない場合に自動作成する安全装置
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ★注意：GitHubにある model.keras が「数字認識用」である必要があります
model = load_model('./model.keras')

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

            # 画像読み込み
            img = image.load_img(filepath, color_mode='grayscale', target_size=(image_size,image_size))
            img = image.img_to_array(img)

            # 正規化
            img = img / 255.0

            # 色の反転チェック
            if np.mean(img) > 0.5:
                img = 1.0 - img
                print("【Info】画像を白黒反転しました", file=sys.stderr) 

            data = np.array([img])
            
            # 予測実行
            result = model.predict(data)[0]
            
            predicted = result.argmax()
            pred_answer = "これは " + classes[predicted] + " です"

            return render_template("index.html",answer=pred_answer)

    return render_template("index.html",answer="")

if __name__ == "__main__":
    # Renderではポート番号を環境変数から取得する必要があります
    port = int(os.environ.get('PORT', 10000))
    app.run(host ='0.0.0.0', port=port)