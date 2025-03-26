#!/bin/bash

# This script creates videos from the recorded frames
# You need to have ffmpeg installed

echo 'Creating episode 3 camera video...'
ffmpeg -y -framerate 20 -pattern_type glob -i 'carla_frames/episode_3/step_*_camera.png' -c:v libx264 -pix_fmt yuv420p carla_frames/episode_3_camera.mp4

echo 'Creating episode 3 lidar video...'
ffmpeg -y -framerate 20 -pattern_type glob -i 'carla_frames/episode_3/step_*_lidar.png' -c:v libx264 -pix_fmt yuv420p carla_frames/episode_3_lidar.mp4

echo 'Creating episode 3 birdeye video...'
ffmpeg -y -framerate 20 -pattern_type glob -i 'carla_frames/episode_3/step_*_birdeye.png' -c:v libx264 -pix_fmt yuv420p carla_frames/episode_3_birdeye.mp4

echo 'Creating episode 3 combined video...'
ffmpeg -y -framerate 20 -pattern_type glob -i 'carla_frames/episode_3/step_*_combined.png' -c:v libx264 -pix_fmt yuv420p carla_frames/episode_3_combined.mp4

