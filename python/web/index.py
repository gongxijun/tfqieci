#_*_ coding: utf-8 _*_

from flask import Flask
from flask import render_template
from flask import request
from flask import Flask, url_for, redirect

from kcwctf_seg import KCWCSeg
from userdicr_jieseg import UserDictSeg

app = Flask(__name__)
kcwc_seg = KCWCSeg()
userdict_seg = UserDictSeg()

submit_text = ""

test_sent = ("李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
    "例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
    "「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"
)


@app.route('/')
def hello_world():
    return redirect(url_for("index"))


@app.route('/jieba_seg_submit', methods=['POST'])
def submitjieba():
    global submit_text
    submit_text = request.form['origin_text_infomation']
    return redirect(url_for("index"))


@app.route('/jieba_seg')
def index():
    text = ""
    global submit_text
    if len(submit_text) > 1:
        text = submit_text
    else:
        text = test_sent.decode("utf-8")
    text = text.strip(" ")
    text = text.strip("\n")
    text = text.strip("\t")
    results = [text, userdict_seg.sentence(text) , kcwc_seg.sentence(text)]
    return render_template('jiebaseg.html', img_url=text, results=results)


if __name__ == '__main__':
    app.run(host='10.87.219.12', port=5112, debug=False)
