"""Matched label-definition ablation on the existing robot-video inputs."""
import json
from pathlib import Path

OPTIONS = ['approach the target object', 'grasp the target object',
           'transport while holding / approach placement destination',
           'open the machine lid', 'place and release the target object',
           'close the machine lid', 'idle / no relevant action',
           'other visible action', 'cannot determine from the supplied input']
BASE = ('The target is the small light-colored coffee capsule initially near the lower center, '
        'not the clear cup on the right. Label the visible action of the gripper interacting '
        'with this capsule or the machine lid. Ignore the other gripper unless it interacts '
        'with the capsule or lid. Times are relative to the supplied input. '
        'The order of listed options is NOT a required workflow. Actions can repeat, be absent, '
        'or occur in a different order. Use only supplied visual evidence. '
        'For a single still image, the motion during every requested interval and the overall '
        'direction are unknown; do not reconstruct a workflow from one frame. '
        'No robot command will be executed. ')
DEFINITIONS = '''Definitions for visible motion (not inferred intention):
1 APPROACH: the empty gripper visibly moves toward the capsule; ends when fingers start closing around it. Mere proximity is insufficient.
2 GRASP: fingers visibly close around the capsule, until closure completes and the capsule begins moving with the gripper. Holding still is not the grasp event.
3 TRANSPORT: capsule is visibly held and moves together with the gripper; includes approaching the machine while holding. Ends when insertion/lowering into the destination begins. No motion while holding is idle.
4 OPEN: machine lid visibly moves from closed toward open. A lid already open, hand proximity, or capsule insertion is not opening.
5 PLACE: held capsule is visibly inserted/lowered onto or into a destination, or fingers visibly open and withdraw leaving it supported there. Mere approach without insertion/lowering is transport. Do not require an off-camera later release.
6 CLOSE: machine lid visibly moves from open toward closed. A lid already closed or hand proximity is not closing.
7 IDLE: no relevant visible movement, including holding still. A missing/occluded target is not evidence of idle.
8 OTHER: a clearly visible relevant motion that satisfies none of 1-7, such as an empty hand withdrawing. Do not force an action into a workflow stage.
9 UNKNOWN: needed motion or target is obscured, the input is a still, or competing actions cannot be resolved.
For each interval choose the action occupying the longest visible duration in that interval, not the most important event. In a tie, or when no action can be established for most of the interval, use 9. If lid motion and capsule motion overlap, label the lid motion (4/6) while it is active; otherwise label capsule motion. Evaluate each interval independently of the option order. Reversed motion must be judged as seen, not as the original intended task.
'''
DIRECTION={'type':'choice','id':'order','question':'Which direction of capsule movement is visually supported across the supplied input?', 'options':['from tabletop toward machine opening','from machine opening toward tabletop','no capsule movement is visible','cannot determine from the supplied input']}
rows=[]
for definition in ['brief','defined']:
 for tag,typ,path in [('forward','video','clips/coffee_forward.mp4'),('reverse','video','clips/coffee_reverse.mp4'),('still','image','frames/last_only.jpg')]:
  intervals=[[i,i+2] for i in range(0,12,2)]
  qs=[{'type':'choice','id':f'p{i}','question':f'What is the dominant visible action during {a}-{b} seconds?', 'options':OPTIONS} for i,(a,b) in enumerate(intervals)]
  rows.append({'id':f'phase_{definition}_{tag}', 'definition':definition,'state':BASE+(DEFINITIONS if definition=='defined' else 'Use the action category names as given.'),'media':[{'type':typ,'path':'docs/video_practical/'+path}],'questions':qs+[DIRECTION], 'intervals_s':intervals,'video_config':{'fps':2,'max_frames':32,'max_video_tokens':768}})
# Same source episode, longer interval: investigate lid opening/closing beyond the original crop.
intervals=[[i,i+2] for i in range(0,22,2)]
rows.append({'id':'phase_defined_full22','definition':'defined','state':BASE+DEFINITIONS,'media':[{'type':'video','path':'docs/video_practical/source/aloha_coffee_ep0_high.mp4'}], 'questions':[{'type':'choice','id':f'p{i}','question':f'What is the dominant visible action during {a}-{b} seconds?', 'options':OPTIONS} for i,(a,b) in enumerate(intervals)]+[DIRECTION], 'intervals_s':intervals, 'video_config':{'fps':2,'max_frames':32,'max_video_tokens':768}})
Path('examples/video_phase_requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
