# Video Stabiliser

Simple cv2 based video stabiliser tool.

## Workflow

1. Video is loaded in memory
2. Run one of available tracking methods
3. Stabilise video by warped affine the frame
4. Render stabilised video

## Tracking and Stabilising

Stabilisation is achieved by warp affining the frame such that tracked pixels coordinate will always be at a fixed
position, constant throughout the video.

From now on that 'fixed position' will be referred as the `pin`.

Tracking can be started at any frame index.
If tracking started at any frame index other than first of the last, tracking will be done in 2 stages

- backwards
- forwards

Then concatenate the tracking results.

Currently available tracking methods:

|         method         | description                                 | keybind | 
|:----------------------:|---------------------------------------------|:-------:|
| optical flow tracking  | Takes mean coordinate of all tracked points |   `o`   |    
|   cv2 tracker method   | Uses TrackerCSRT coordinate                 |   `t`   |    
| color masking tracking | Takes mean coordinate of color mask         |   `c`   |    

Upon choosing `cv2 tracker method` or `optical flow tracking`, user must select ROI bounding box of the region to be
tracked, then press `enter` or `space`.

Choosing `color masking tracking` will create a new window for user to set a color mask by changing the min and max HSV
values
with their respective sliders. Changing frame index is also available in the new window.

### Controls for color masking viewer

| keybind  | description                            |
|:--------:|----------------------------------------|
| `enter`  | Apply color masking                    |   
| `space`  | Apply color masking                    |   
|   `r`    | reset hsv min and max value to default |    
| `escape` | Exit color masking viewer              |    
|   `q`    | Exit color masking viewer              |    

### Setting of "the pin"

Once tracking is done, pin is automatically set at the frame index where tracking started.

To change the pin at different frame index, scroll to desired frame index and set pin by pressing `p`, it will override
previous pin coordinate with the tracked coordinate at the respective frame index.

To unset the pin, press `u`.

Frames will not be stabilised but tracking curve will be displayed on the frame.

### Manual set of the pin

To turn on manual pinning pres `m`.

Now you can move the pin with:

| keybind | direction |
|:-------:|:---------:|
|   `w`   |    up     |
|   `a`   |   left    |
|   `s`   |   down    |
|   `d`   |   right   |

The pin will move 1 pixel at a key press.

> In order to use 'wasd' the toggle manual pin setting must be on (`m` should be press), else it will not work.
> This is to exclude accidental moving the pin.

### Locking axis

Axis can be locked by changing `lock_axis` slider value.

| value | slider position | description                                                                  |
|:-----:|:---------------:|------------------------------------------------------------------------------|
|   0   |  left position  | lock vertical position, only horizontal movement for the pin will be allowed |
|   1   | center position | tracking is not constrained                                                  |
|   2   | right position  | lock horizontal position, only vertical movement for the pin will be allowed |

### Smoothing tracking results

Smoothing tracking results is controlled by `conv_smooth` slider.

Smoothing effect is achieved with convolving the tracked data.

Value of `conv_smooth` slider represents window_size for convolution.

## Rendering

Rendering will be done following the settings from the control windows.

To render stabilised video press `r`.

Pressing `R` will iterate trough axis lock options and render 3 separate videos:

- vertical locked
- unlocked
- horizontal locked

> If a video with same settings has already been rendered, rendering again will overwrite previous video.

## Controls summary (windows)

Currently all the keybinds are windows.
Return of`cv2.waitKey()` is OS dependent.

|  key   | function                     | additional description                                                    |
|:------:|------------------------------|---------------------------------------------------------------------------|
| escape | go to the next video         |                                                                           |
|   q    | go to the next video         |                                                                           |
|   Q    | go to the next video         |                                                                           |
|   +    | rescale_larger               | Make window containing current frame larger                               |
|   -    | rescale_smaller              | Make window containing current frame smaller                              |
|   p    | set_pin                      | Sets pin at the current frame index                                       |
|   u    | set_pin                      | Doesn't warp affine the frame                                             |
|   m    | toggle_manual_pin            | Allows manually pinning ('wasd' keys)                                     |
|   C    | toggle_cross                 | Displays crosshair on the frame. for visual reference of the frame center |
|   o    | start_optical_flow_tracking  | Tracking with **optical flow** method                                     |
|   c    | start_color_masking_tracking | Tracking with **color masking** method                                    |
|   t    | start_tracker_method         | Tracking with cv2 build in **TrackerCSRT**                                |
|   r    | render_op                    | Renders video with applied stabilisation                                  |
|   R    | render_all_axis              | Renders 3 videos for all exis lock variants                               |