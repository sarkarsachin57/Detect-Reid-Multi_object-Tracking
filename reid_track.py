
from init import *
import argparse

def ReidTrack(input_path, is_stream):

    if is_stream:
        stream = VideoStream(input_path).start()

    else:
        video = cv2.VideoCapture(input_path)
        video_fps = video.get(cv2.CAP_PROP_FPS)


    frame_no = 0
    track_targets = []

    while True:

        if is_stream:
            frame = stream.read()

        else:
            _, frame = video.read()

        
        if frame is None:
            print("No frame found! Stoping the pipeline.")
            break


        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        dets = detect(rgb).cpu().detach().numpy()

        dets_filtered = []    
        croped_images = []

        for det in dets:
            x1, y1, x2, y2, conf, cls = det
            startX, startY, endX, endY = int(x1), int(y1), int(x2), int(y2)
            class_name = class_names[int(cls)]
            if conf > 0.45 and class_name in ['person']:
                dets_filtered.append([startX, startY, endX, endY, class_name, round(conf, 2)])
                croped = rgb[startY:endY,startX:endX]
                croped_images.append(croped)


        if len(croped_images) != 0:

            features_curr = extractor(croped_images)


            if frame_no == 0 or track_targets == []:

                track_targets = features_curr



            else:

                sim_res = 1-torchreid.metrics.compute_distance_matrix(features_curr,track_targets,'cosine').cpu().detach().numpy()

                query_id_used, gallery_id_used = [], []
                for sim in -np.sort(-np.concatenate(sim_res)):
                    sim_ids = np.array(np.where(sim_res == sim)).T
                    for query_id, gallery_id in sim_ids:
                        startX, startY, endX, endY, class_name, conf = dets_filtered[query_id]
                        if query_id not in query_id_used and gallery_id not in gallery_id_used and sim > 0.60:
                            
                            cv2.rectangle(frame, (startX, startY), (endX, endY),get_mot_color(idx=gallery_id+1), 2)
                            draw_bb_text(frame, f'{class_name}, ID : {gallery_id}', (startX, startY, endX, endY),cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 0), 1, get_mot_color(idx=gallery_id+1))
                           
                            if sim > 0.65:
                                track_targets[gallery_id] = features_curr[query_id]

                            query_id_used.append(query_id), gallery_id_used.append(gallery_id)

                        elif query_id not in query_id_used and sim_res[query_id].max() < 0.55:
                            track_targets = torch.cat([track_targets, torch.tensor([features_curr[query_id].tolist()]).to(device)])
                            query_id_used.append(query_id)
                            id = len(track_targets) - 1



        frame_no += 1


            
        cv2.imshow(f'DISPLAYING REID-TRACKING OF {input_path}', frame)

        if is_stream:
            key = cv2.waitKey(1)
        else:
            key = cv2.waitKey(int(1000//video_fps))

        if key is ord('q'):
            break


    if is_stream:
        stream.stop()
    else:
        video.release()

    cv2.destroyWindow(f'DISPLAYING REID-TRACKING OF {input_path}')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
                    prog='ReidTrack',
                    description='Multi-Object Tracking using detection and reidentification in every frame.')
    

    parser.add_argument('--input-path', required=True)      
    parser.add_argument('--is-stream', action='store_true')


    args = parser.parse_args()


    ReidTrack(input_path = args.input_path, is_stream = args.is_stream)


























































































