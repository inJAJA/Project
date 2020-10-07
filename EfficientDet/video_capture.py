import os
import cv2

# Class that allows continuous capture of the semantic points of the image
project = 'soccer_ball_data'
video_name = 'train02.mp4'

cap = cv2.VideoCapture(os.path.join('datasets', f'{project}','sample_video', video_name))

if video_name.endswith('mp4'):
    name = video_name.replace('.mp4', '')

elif video_name.endswith('avi'):
    name = video_name.replace('avi', '')
else:
    name = video_name

def getFrame(sec):
    cap.set(cv2.CAP_PROP_POS_MSEC, 10000 + sec * 1000)   # cv2.CAP_PROP_POS_MSEC :  동영상 파일의 프레임 위치(MS)
    hasFrames, image = cap.read()
    if hasFrames:
        os.makedirs(f'datasets/{project}/captures', exist_ok=True)
        cv2.imwrite( f'datasets/{project}/captures/' + f'{name}' + str(count) + ".jpg", image)        # save frame as JPG file

    return hasFrames

sec = 0
frameRate = 0.5
count = 1
success = getFrame(sec)
while sec < 2000:
    count += 1
    sec += frameRate
    sec = round(sec, 2)
    success = getFrame(sec)