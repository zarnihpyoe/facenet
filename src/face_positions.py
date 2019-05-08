import os
import numpy as np
import facenet

def save_position_into_file(src, dest):
    # storing bounding boxes' centers with their respective images name
    d = {}
    with open(src) as f:
        for line in f:
            line = line.split()
            line[0] = line[0].split('/')[-1]
            d[line[0]] = (
                ( ( (int(line[1])+int(line[3]))//2 ) / (1280//2) ) - 1, 
                ( ( (int(line[2])+int(line[4]))//2 ) / (720//2) ) - 1
            )

    # getting name of images in order
    # files = [os.path.join(dest, f) for f in os.listdir(dest)]
    # files.sort()
    # names = []
    # for p in [f for f in files if os.path.isdir(os.path.join(dest, f))]:
    #     img_names = os.listdir(p)
    #     img_names.sort()
    #     names.extend(img_names)
    
    datadir = '/Users/zarnihpyoe/wpi/mqp/data5/classified'
    dataset = facenet.get_dataset(datadir)
    paths, _ = facenet.get_image_paths_and_labels(dataset)
    names = [path.split('/')[-1] for path in paths]
    # making positions matrix (m, 2)
    embeddings, labels = np.load('{}/embeddings.npy'.format(dest))
    print(embeddings.shape)
    positions = np.zeros((embeddings.shape[1], 2))

    for i, n in enumerate(names):
        c = d[n]
        positions[i] = [c[0],c[1]]
    
    # appending positions matrix to the end of the embeddings (512+2, m)
    # ext_embeddings = np.vstack((positions.T, embeddings))
    # print(ext_embeddings.shape, len(labels))
    # np.save(os.path.join(dest, 'ext_embeddings.npy'), ext_embeddings)
    # np.save(os.path.join(dest, 'labels.npy'), labels)

    # just saving the positions of the detected faces
    print(positions.shape)
    np.save(os.path.join(dest, 'face_positions.npy'), positions.T)


# save_position_into_file(
#     '/Users/zarnihpyoe/wpi/mqp/data3/test_faces/bounding_boxes_25601.txt',
#     '/Users/zarnihpyoe/wpi/mqp/data3/used_faces_maj'
# )
# save_position_into_file(
#     '/Users/zarnihpyoe/wpi/mqp/data/0.3/bounding_boxes_88855.txt',
#     '/Users/zarnihpyoe/wpi/mqp/data/0.3'
# )

# don't forget to change the size of the frame in the normalization formula
# save_position_into_file(
#     '/Users/zarnihpyoe/wpi/mqp/data4/0.1/bounding_boxes_36060.txt',
#     '/Users/zarnihpyoe/wpi/mqp/data4/0.1'
# )

# don't forget to change the size of the frame in the normalization formula
save_position_into_file(
    '/Users/zarnihpyoe/wpi/mqp/data5/aligned/bounding_boxes_83103.txt',
    '/Users/zarnihpyoe/wpi/mqp/data5'
)