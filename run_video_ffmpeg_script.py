import subprocess
import os
import platform
import shlex

def run_ffplay(video_path, width, height,start_frame=None, end_frame=None, fps=30, slow_factor=1.0):
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return
    
    start_sec = start_frame / fps if start_frame is not None else None
    duration = ((end_frame - start_frame) / fps) if start_frame is not None and end_frame is not None else None
    # Build ffplay command
    cmd = ['ffplay', video_path]
    if width and height:
        cmd += ['-x', str(width), '-y', str(height)]
   
    # Build ffplay command
    cmd = ['ffplay', '-autoexit']
    
    vf_filters = []
    
    # Slow motion
    if slow_factor > 1.0:
        cmd += ['-vf', f'setpts={slow_factor}*PTS']

    # Frame-based timing
    if start_sec is not None:
        cmd += ['-ss', str(start_sec)]
    if duration is not None:
        cmd += ['-t', str(duration)]

    # Window size
    if width and height:
        cmd += ['-x', str(width), '-y', str(height)]

    cmd.append(video_path)

    try:
        subprocess.run(cmd)
    except FileNotFoundError:
        print("ffplay not found. Make sure FFmpeg is installed and in your system PATH.")
    except Exception as e:
        print(f"Error: {e}")