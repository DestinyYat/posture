# from flask import Flask, request, jsonify, send_from_directory
# from werkzeug.utils import secure_filename
# from dt123 import analyze_image
# import os

# app = Flask(__name__)
# UPLOAD_FOLDER = 'uploads'
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image uploaded'}), 400

#     file = request.files['image']
#     if file.filename == '':
#         return jsonify({'error': 'Empty filename'}), 400
#     if not allowed_file(file.filename):
#         return jsonify({'error': 'File type not allowed'}), 400

#     filename = secure_filename(file.filename)
#     save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(save_path)

#     try:
#         result = analyze_image(save_path)
#         return jsonify({
#             "status": "success",
#             "filename": filename,
#             "diagnosis": result["diagnosis"],
#             "plot_url": f"/results/{os.path.basename(result['plot_path'])}"
#         })
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/results/<filename>')
# def serve_result_image(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# if __name__ == '__main__':
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#     app.run(host='0.0.0.0', port=5000, debug=True)




from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from dt123 import analyze_image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    try:
        result = analyze_image(save_path)
        return jsonify({
            "status": "success",
            "filename": filename,
            "diagnosis": result["diagnosis"],
            "plot_url": f"/results/{os.path.basename(result['plot_path'])}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def serve_result_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
@app.route('/rewq')
def serveimge():
    return "<h1>Hello, World!</h1><p><a href='/form'>去反馈页面</a></p>"


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=80, debug=True)