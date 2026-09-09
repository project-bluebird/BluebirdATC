import os

import imageio


def generate_video(
    render_dir: str,
    frame_prefix: str,
    video_filename: str = "eval",
    clean_up: bool = True,
):
    # get frames
    filenames = os.listdir(render_dir)
    filenames = [fn for fn in filenames if fn.startswith(frame_prefix)]
    filenames = [fn for fn in filenames if fn.endswith(".png")]
    filenames = sorted(filenames, key=lambda fn: (os.path.splitext(fn)[0]).split("_")[1])

    images = []
    for fn in filenames:
        frame_path = os.path.join(render_dir, fn)
        images.append(imageio.v3.imread(frame_path))

    # save render video (gif composed of the frames)
    imageio.mimsave(os.path.join(render_dir, f"{video_filename}.gif"), images, loop=0)

    if clean_up:
        # delete individual frames used to create the render video
        for fn in filenames:
            os.remove(os.path.join(render_dir, fn))
    return
