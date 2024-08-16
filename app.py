import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
# 用户上传的文件的保存路径
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


def get_fish_description(fish_label):
    descriptions = {
        'Abactochromis_labrosus': '厚唇黑慈鱼，为辐鳍鱼纲鲈形目隆头鱼亚目慈鲷科的其中一种，分布于非洲马拉维湖流域，为特有种，栖息深度可达30米，体长可达11.5公分，生活在岩石底部水域，生活习性不明，可作为观赏鱼。',
        'Abalistes_stellaris': '星点宽尾鳞鲀（学名：Abalistes stellaris）为辐鳍鱼纲鲀形目鳞鲀亚目鳞鲀科的其中一种，分布于印度西太平洋区，从红海、东非至东南亚，北起日本，南迄澳洲；东大西洋南非及圣赫勒拿岛海域，栖息深度61-180米，本鱼体椭圆，尾鳍上下叶延长，鳞片在胸鳍基部逐渐增大，且在鳃缝正后方形成一个柔韧有弹性的鼓膜，在后部身体有的鳞片突出形成纵向的脊，背鳍硬棘3枚，背鳍软条25-27枚，臀鳍软条24-26枚，体长可达60公分，栖息在沿岸沙泥底质、藻类、海绵生长的水域，以底栖无脊椎动物为食，生活习性不明，可作为食用鱼及观赏鱼。',
        'Ablabys_taenianotus': '帆鳍鲉（学名：Ablabys taenianotus），又称背带长绒鲉，俗名钝顶鲉、济公，为辐鳍鱼纲鲉形目真裸皮鲉科的其中一种。本鱼分布于印度西太平洋区，包括安达曼海、斐济、台湾、日本、澳洲等海域。本鱼体延长，侧扁。头短钝，背缘近垂直或略凹，上方无棘棱。齿绒毛状，上下颌、锄骨及腭骨均具齿带。体被细小圆鳞，埋于皮下。背鳍连续，起点在眼前上方，鳍间无缺刻，最后鳍条一部分与尾鳍相连。体呈浅褐色到深褐色，鼻及前额有白斑，体长不超过10公分。常栖息于礁穴内，有时亦出现于潮池或碎砾石堆上，会躺在礁石上模拟成枯叶状，伺机捕食小鱼或甲壳类，亦可迷惑敌人，以避免危险。各鳍鳍棘亦具毒性。无食用价值，一般皆做下杂鱼，偶会在水族馆看到。',
        'Ablennes_hians': '横带扁颌针鱼（学名：Ablennes hians），又称扁鹤鱵，俗名鲎鱼、青旗、学仔、白天青旗，为辐鳍鱼纲鹤鱵目鹤鱵亚目鹤鱵科的其中一种。本鱼分布于全世界的热带与温带的水域。本鱼体甚侧扁，略成带状，截面圆楔型，体高为体宽的2至3倍；两颚突出如长喙，具带状排列之细齿，且具一行稀疏排列之大犬齿；锄骨无齿；头背部平扁，头盖骨背侧之中央沟发育不良；主上腭骨之下缘于嘴角处完全被眼前骨所覆盖；尾柄侧扁，其高远小于其宽，无侧隆起棱；背鳍1枚，与臀鳍对在于体之后方，臀鳍鳍条数多于背鳍，背鳍据23至25枚软条，臀鳍具25至27枚软条，背鳍起点在臀鳍第5至7软条基底之上方，两者之前方鳍条均延长成镰刀状，此外背鳍后方数鳍条亦略延长；腹鳍基底位于眼前缘与尾鳍基底间之中央略前；尾鳍深开叉，下叶长于上叶；鳞细小，无鳃耙。体背翠绿色至暗绿色，腹部银白色；体侧具8至13条暗蓝色横带；各鳍淡翠绿色，边缘黑色；两腭齿亦呈绿色。体长可达140公分。为表层游泳鱼类，常嬉戏于河口域，常会因追逐猎物而跳出水面，因其布满锐利牙齿之长喙状上下颚，极为危险，常有被伤之报导，而有水面杀手之称。产卵时会进入海水与半淡咸水交汇处，于近岸之海藻下产卵，一次产卵数千，其卵具缠络丝。可食用，其市场价格不高，尤其在夏季时，其腹部中有许多白色细长的寄生虫，料理时须加煮熟。'
    }
    return descriptions.get(fish_label, '未找到介绍信息')



def classify_fish(image_file):
    export_path = 'model'
    model = tf.saved_model.load(export_path)
    class_labels = ['Abactochromis_labrosus', 'Abalistes_stellaris', 'Ablabys_taenianotus', 'Ablennes_hians']

    img = image.load_img(image_file, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    inference = model.signatures["serving_default"]
    input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    output = inference(input_tensor)
    predictions = output['output_0']
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_label = class_labels[predicted_class_index]

    fish_description = get_fish_description(predicted_class_label)

    return predicted_class_label, image_file, fish_description


@app.route('/upload', methods=["POST"])
def upload():
    file = request.files["image"]
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        result, image_url, fish_description = classify_fish(file_path)
        return render_template('index.html', result=result, statement="当前文件：" + filename, image_url=image_url, fish_description=fish_description)
    else:
        return render_template('index.html', statement="上传文件无效，请确保上传的文件是图片文件且格式为jpg、jpeg、png或gif。")


if __name__ == '__main__':
    app.secret_key = 'supersecretkey'
    app.config['SESSION_TYPE'] = 'filesystem'
    # 只允许用户上传图片文件
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.run(debug=True)
