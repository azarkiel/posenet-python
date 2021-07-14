import tensorflow as tf
import cv2
# import time
import argparse
import posenet

import getopt, sys
import time, os, psutil
import Metrikas as met
import logging


def main(argv):
    mi_logger = met.prepareLog('Log_' + sys.argv[0] + '.log', logging.INFO)
    process = psutil.Process(os.getpid())

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=int, default=101)
    parser.add_argument('--cam_id', type=int, default=0)
    parser.add_argument('--cam_width', type=int, default=1280)
    parser.add_argument('--cam_height', type=int, default=720)
    parser.add_argument('--scale_factor', type=float, default=0.7125)
    parser.add_argument('--file', '--inputfile', type=str, default=None, help="Optionally use a video file instead of a live camera")
    parser.add_argument('--printmetrics', action="store_true", help='Se agregan las metricas a la imagen de salida')
    parser.add_argument('--noview', action="store_true", help='Se procesa el video sin mostrarlo')

    args = parser.parse_args()

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        cap = cv2.VideoCapture(args.file if args.file else args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        p_time = 0
        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_actual = 0
        fps_list = []
        cpu_list = []
        mem_list = []

        while True:
            frame_actual += 1

            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image})

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            keypoint_coords *= output_scale

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            c_time = time.time()
            fps = 1 / (c_time - p_time)
            p_time = c_time
            fps_list.append(fps)

            if not args.noview:
                if args.printmetrics:
                    # cv2.imshow('posenet', overlay_image)
                    cv2.imshow('posenet', met.printMetrics(overlay_image, frame_actual, num_frames, fps, round(met.Average(fps_list), 1), round(max(fps_list), 1)))
                else:
                    cv2.imshow('posenet', overlay_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            mem_list.append(process.memory_info()[0])
            cpu_porcentaje = round(process.cpu_percent() / psutil.cpu_count(), 1)
            cpu_list.append(cpu_porcentaje)

            sys.stdout.write("\rFrame " + str(frame_actual) + "/" + str(int(num_frames)) + " " + str(
                round(100 * frame_actual / num_frames, 1)) + "%"
                             + "\tUsedMemory=" + str(round(process.memory_info()[0] / (1024 ** 2), 1)) + "MB"
                             + "\tUsedCPU=" + str(cpu_porcentaje) + "%")
            sys.stdout.flush()

        fps_list = [i for i in fps_list if i > 0.5]
        mem_list = [i for i in mem_list if i > 0.5]
        cpu_list = [i for i in cpu_list if i > 0.5]
        resumen = ('\nARGS=' + ' '.join(str(e) for e in sys.argv)
                   # + '\nPROGRAM= ' + sys.argv[0]
                   # + '\nFILENAME= ' + sys.argv[2]
                   + '\nFPS_AVG= ' + str(round(met.Average(fps_list), 1))
                   + '\nFPS_MAX= ' + str(round(max(fps_list), 1))
                   + '\nFPS_MIN= ' + str(round(min(fps_list), 1))
                   + '\nMEM_AVG= ' + str(round(met.Average(mem_list) / (1024 ** 2), 1)) + 'MB'  # in bytes
                   + '\nMEM_MAX= ' + str(round(max(mem_list) / (1024 ** 2), 1)) + 'MB'  # in bytes
                   + '\nMEM_MIN= ' + str(round(min(mem_list) / (1024 ** 2), 1)) + 'MB'  # in bytes
                   + '\nCPU_AVG= ' + str(round(met.Average(cpu_list), 1)) + '%'
                   + '\nCPU_MAX= ' + str(round(max(cpu_list), 1)) + '%'
                   + '\nCPU_MIN= ' + str(round(min(cpu_list), 1)) + '%'
                   + '\n')
        print(resumen)
        mi_logger.info(resumen)
        # print('Average FPS: ', frame_actual / (time.time() - start))


if __name__ == "__main__":
    main(sys.argv[1:])
