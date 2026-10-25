"""Prepare an auditable short video, reverse and still controls for manual trials."""
import json
from pathlib import Path
import av

BASE = Path('docs/video_practical')
SOURCE = BASE / 'source/aloha_coffee_ep0_high.mp4'
FPS = 10
FRAMES = 128

def encode(path, images):
    with av.open(str(path), 'w') as out:
        stream = out.add_stream('libx264', rate=FPS)
        stream.width, stream.height = images[0].size
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '23', 'preset': 'fast'}
        for image in images:
            frame = av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)

images=[]
with av.open(str(SOURCE)) as source:
    for frame in source.decode(video=0):
        if float(frame.time or 0) + 1e-6 >= len(images) / FPS:
            images.append(frame.to_image())
        if len(images) == FRAMES:
            break
assert len(images) == FRAMES
encode(BASE / 'clips/coffee_forward.mp4', images)
encode(BASE / 'clips/coffee_reverse.mp4', list(reversed(images)))
images[-1].save(BASE / 'frames/last_only.jpg')
images[0].save(BASE / 'frames/reference_capsule.jpg')

choices=['on the tabletop and not visibly held',
         'visibly in a gripper away from the machine opening',
         'at or inside the machine opening',
         'not visible or cannot determine']
state=('The input is a robot coffee-preparation clip, or a still-image control. '
       'The target is the small round light-colored coffee capsule initially near the lower center, '
       'not the clear drinking cup on the right. Time is relative to the supplied clip, starting at 0 seconds. '
       'For each requested time, report only visible capsule state near that time; use cannot determine when obscured. '
       'If the input is a single still image, do not infer earlier time points or movement order. '
       'This is an offline annotation request; no robot command will be executed.')
def questions(times, include_direction=True):
    qs=[{'type':'choice','id':f'p{i}',
         'question':f'At approximately {t:g} seconds in this input, where is the target capsule?',
         'options':choices} for i,t in enumerate(times)]
    if include_direction:
        qs.append({'type':'choice','id':'order',
                   'question':'What temporal sequence is visibly supported for the capsule across the supplied input?',
                   'options':['from tabletop toward the machine opening',
                              'from the machine opening toward the tabletop',
                              'no capsule movement is visible',
                              'cannot determine from the supplied input']})
    return qs
rows=[]
def add(id,path,typ,times,tokens=768):
    rows.append({'id':id,'use_case':'作業動画の時間窓アノテーション',
                 'state':state,'media':[{'type':typ,'path':path}],
                 'questions':questions(times),'video_config':{'fps':2,'max_frames':32,'max_video_tokens':tokens},
                 'query_times_s':times})
forward=str(BASE / 'clips/coffee_forward.mp4')
add('video_forward_6',forward,'video',[1,3,5,7,9,11])
add('video_reverse_6',str(BASE/'clips/coffee_reverse.mp4'),'video',[1,3,5,7,9,11])
add('still_last_6',str(BASE/'frames/last_only.jpg'),'image',[1,3,5,7,9,11])
add('video_forward_3',forward,'video',[1,7,11])
add('video_forward_12',forward,'video',[.5+i for i in range(12)])
add('video_forward_6_higher_resolution',forward,'video',[1,3,5,7,9,11],1536)
Path('examples/video_practical_requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
(BASE/'preparation.json').write_text(json.dumps({'source':str(SOURCE),'source_interval_s':[0,12.8],
    'output_fps':FPS,'output_frames':FRAMES,'forward':'coffee_forward.mp4','reverse':'frame-reversed forward clip',
    'still_control':'last decoded frame of forward clip','images_size':images[0].size,
    'annotation_status':'preliminary labels to be inspected; no independent human ground truth'},indent=2))
print('Prepared forward/reverse MP4s, still control and',len(rows),'requests')
