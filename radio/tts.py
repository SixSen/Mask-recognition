from aip import AipSpeech

#""" 你的 APPID AK SK """
APP_ID = '18081093'
API_KEY = 'Qv4q0sKRNYLniDgUdAHd5iYP'
SECRET_KEY = 'tX0SYY01H8sR3jhcl4Uvq7zwsQuHYo9e'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
def tts_main(words):
    result  = client.synthesis(words, 'zh', 1, {
        'vol': 5,
    })

    # 识别正确返回语音二进制 错误则返回dict 参照下面错误码
    if not isinstance(result, dict):
        with open('slogan.wav', 'wb') as f:
            f.write(result)

tts_main('野味不要尝，以免损健康。')
#tts_main('请您戴好口罩')
