import os
from flask import Flask, request

UPLOAD_FOLDER = "./upload"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def upload_file():
    return """
    <h1>Upload new File</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file1">
      <input type="submit">
    </form>
    """



@app.route("/dtr", methods=["GET", "POST"])
def dtr():
    os.system("python3 dtr.py")
    return "dtr done"

@app.route("/adaboost", methods=["GET", "POST"])
def adaboost():
    os.system("python3 adaboost.py")
    return "adaboost done"

@app.route("/svr", methods=["GET", "POST"])
def svr():
    os.system("python3 svr.py")
    return "svr done"


if __name__ == "_main_":
    # app.run()
    app.run(host="0.0.0.0", port=5000)