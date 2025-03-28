import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lightglue import LightGlue, SIFT
from lightglue.utils import rbd
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from ultralytics import YOLO


class ImageProcessor:
    def __init__(self, clean_image_path = None, dirty_image_path = None, sam_model_path = "models/sam_vit_h_4b8939.pth", yolo_model_path = "models/yolov10x.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load images
        # self.clean_image_path = clean_image_path
        # self.dirty_image_path = dirty_image_path
        # self.clean_image = cv2.imread(clean_image_path)
        # self.dirty_image = cv2.imread(dirty_image_path)

        # Initialize feature extractor and matcher
        self.extractor = SIFT()
        self.matcher = LightGlue(features="sift").eval().to(self.device)

        # Load SAM model
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_model_path).to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

        # Load YOLO model
        self.yolo_model = YOLO(yolo_model_path).to(self.device)
        self.objects = 0

    def warp_dirty_to_clean(self):
        """Aligns the dirty image to the clean image using SIFT + LightGlue feature matching."""
        gray_clean = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2GRAY)
        gray_dirty = cv2.cvtColor(self.dirty_image, cv2.COLOR_BGR2GRAY)

        image0 = torch.tensor(gray_clean, dtype=torch.float32, device=self.device) / 255.0
        image1 = torch.tensor(gray_dirty, dtype=torch.float32, device=self.device) / 255.0

        image0 = image0.unsqueeze(0).unsqueeze(0)
        image1 = image1.unsqueeze(0).unsqueeze(0)

        # Extract features
        with torch.no_grad():
            feats0 = self.extractor.extract(image0)
            feats1 = self.extractor.extract(image1)

        # Match features using LightGlue
        matches01 = self.matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

        # Extract matched keypoints
        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        pts_clean = kpts0[matches[..., 0]].cpu().numpy().astype(np.float32)
        pts_dirty = kpts1[matches[..., 1]].cpu().numpy().astype(np.float32)

        # Find homography
        homography_matrix, _ = cv2.findHomography(pts_dirty, pts_clean, cv2.RANSAC, 5.0)

        # Warp dirty image to align with clean image
        aligned_dirty_image = cv2.warpPerspective(self.dirty_image, homography_matrix,
                                                  (self.clean_image.shape[1], self.clean_image.shape[0]))

        self.dirty_image = aligned_dirty_image  # Update dirty image
        return aligned_dirty_image

    def segment_and_detect(self):
        """Applies SAM for segmentation and YOLO for object detection on the aligned image."""
        clean_rgb = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2RGB)
        dirty_rgb = cv2.cvtColor(self.dirty_image, cv2.COLOR_BGR2RGB)

        # Generate segmentation masks
        with torch.no_grad():
            clean_masks = self.mask_generator.generate(clean_rgb)
            dirty_masks = self.mask_generator.generate(dirty_rgb)

        # Detect objects
            clean_objects = self.detect_objects(clean_rgb, clean_masks, 'clean')
            dirty_objects = self.detect_objects(dirty_rgb, dirty_masks, 'dirty')

        return self.compare_objects(clean_objects, dirty_objects)

    def detect_objects(self, image, masks, category):
        """Detects objects in segmented image regions using YOLO."""
        detected_objects = []
        for mask in masks:
            segmentation = mask['segmentation']
            x, y, w, h = cv2.boundingRect(segmentation.astype(np.uint8))
            cropped_object = image[y:y+h, x:x+w].copy()
            cv2.imwrite(f'objects/{category}_{self.objects}.jpg', cropped_object)
            self.objects += 1

            # Run YOLO detection
            results = self.yolo_model(cropped_object)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    label = self.yolo_model.names[cls]

                    # Map back to original image coordinates
                    x1, x2 = x + x1, x + x2
                    y1, y2 = y + y1, y + y2

                    detected_objects.append((label, (x1, y1, x2, y2)))
        return detected_objects

    def compare_objects(self, clean_objs, dirty_objs, threshold=200):
        """Compares objects between clean and dirty images to find new, displaced, or removed objects."""
        from scipy.spatial import distance

        displaced_objects, removed_objects, new_objects = [], [], []
        clean_dict = {label: [] for label, _ in clean_objs}
        for label, bbox in clean_objs:
            clean_dict[label].append(bbox)

        dirty_dict = {label: [] for label, _ in dirty_objs}
        for label, bbox in dirty_objs:
            dirty_dict[label].append(bbox)

        # Find displaced and removed objects
        for label, clean_bbox_list in clean_dict.items():
            for clean_bbox in clean_bbox_list:
                found, min_dist, closest_bbox = False, float("inf"), None

                if label in dirty_dict:
                    for dirty_bbox in dirty_dict[label]:
                        dist = distance.euclidean(
                            [(clean_bbox[0] + clean_bbox[2]) / 2, (clean_bbox[1] + clean_bbox[3]) / 2],
                            [(dirty_bbox[0] + dirty_bbox[2]) / 2, (dirty_bbox[1] + dirty_bbox[3]) / 2]
                        )
                        if dist < threshold and dist < min_dist:
                            min_dist, closest_bbox, found = dist, dirty_bbox, True

                if found and min_dist > 10:
                    displaced_objects.append((label, clean_bbox, closest_bbox))
                elif not found:
                    removed_objects.append((label, clean_bbox))

        # Find new objects
        for label, dirty_bbox_list in dirty_dict.items():
            for dirty_bbox in dirty_bbox_list:
                found = any(
                    distance.euclidean(
                        [(clean_bbox[0] + clean_bbox[2]) / 2, (clean_bbox[1] + clean_bbox[3]) / 2],
                        [(dirty_bbox[0] + dirty_bbox[2]) / 2, (dirty_bbox[1] + dirty_bbox[3]) / 2]
                    ) < threshold
                    for clean_bbox in clean_dict.get(label, [])
                )
                if not found:
                    new_objects.append((label, dirty_bbox))

        return displaced_objects, removed_objects, new_objects
    
    # def draw_results(self, displaced, removed, new):
    #     """Draws the results on the dirty image."""
    #     dirty_image_draw = self.dirty_image.copy()

    #     for label, bbox in new:
    #         x1, y1, x2, y2 = bbox
    #         if x2 - x1 < 200 and y2 - y1 < 200:
    #             continue

    #         cv2.imwrite(f'objects/{label}.jpg', dirty_image_draw[y1:y2, x1:x2])
    #         cv2.rectangle(dirty_image_draw, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red for new objects
    #         cv2.putText(dirty_image_draw, f"New: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    #     # for label, _, dirty_bbox in displaced:
    #     #     x1, y1, x2, y2 = dirty_bbox
    #     #     cv2.rectangle(dirty_image_draw, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue for displaced objects

    #     # for label, bbox in removed:
    #     #     x1, y1, x2, y2 = bbox
    #     #     cv2.rectangle(dirty_image_draw, (x1, y1), (x2, y2), (0, 0, 0), 3)  # Black for removed objects

    #     cv2.imwrite('result.jpg', dirty_image_draw)
    #     plt.figure(figsize=(12, 6))
    #     plt.imshow(cv2.cvtColor(dirty_image_draw, cv2.COLOR_BGR2RGB))
    #     plt.axis("off")
    #     plt.title("Detected Changes")
    #     plt.show()

    def draw_results(self, displaced, removed, new):
        """Draws the results on the dirty image with merged overlapping objects."""
        dirty_image_draw = self.dirty_image.copy()
        image_list = []
        label_list = []
        # def compute_iou(box1, box2):
        #     """Computes Intersection over Union (IoU) between two bounding boxes."""
        #     x1, y1, x2, y2 = box1
        #     x1_, y1_, x2_, y2_ = box2

        #     # Compute intersection
        #     inter_x1 = max(x1, x1_)
        #     inter_y1 = max(y1, y1_)
        #     inter_x2 = min(x2, x2_)
        #     inter_y2 = min(y2, y2_)

        #     inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        #     # Compute union
        #     box1_area = (x2 - x1) * (y2 - y1)
        #     box2_area = (x2_ - x1_) * (y2_ - y1_)
        #     union_area = box1_area + box2_area - inter_area

        #     return inter_area / union_area if union_area > 0 else 0

        # def merge_bounding_boxes(objects, iou_threshold=0.5):
        #     """Merges overlapping bounding boxes based on IoU threshold."""
        #     merged_objects = []
        #     while objects:
        #         label, base_box = objects.pop(0)
        #         x1, y1, x2, y2 = base_box
        #         to_merge = [(label, base_box)]

        #         for other_label, other_box in objects[:]:
        #             if compute_iou(base_box, other_box) >= iou_threshold:
        #                 to_merge.append((other_label, other_box))
        #                 objects.remove((other_label, other_box))

        #         # Merge bounding boxes
        #         merged_x1 = min(box[1][0] for box in to_merge)
        #         merged_y1 = min(box[1][1] for box in to_merge)
        #         merged_x2 = max(box[1][2] for box in to_merge)
        #         merged_y2 = max(box[1][3] for box in to_merge)

        #         merged_objects.append((label, (merged_x1, merged_y1, merged_x2, merged_y2)))

        #     return merged_objects

        # def merge_bounding_boxes(objects, iou_threshold=0.5):
        #     """Merges all overlapping and nested bounding boxes into larger ones."""
            
        #     def compute_iou(box1, box2):
        #         """Computes Intersection over Union (IoU) between two bounding boxes."""
        #         x1, y1, x2, y2 = box1
        #         x1_, y1_, x2_, y2_ = box2

        #         # Compute intersection
        #         inter_x1 = max(x1, x1_)
        #         inter_y1 = max(y1, y1_)
        #         inter_x2 = min(x2, x2_)
        #         inter_y2 = min(y2, y2_)

        #         inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        #         # Compute union
        #         box1_area = (x2 - x1) * (y2 - y1)
        #         box2_area = (x2_ - x1_) * (y2_ - y1_)
        #         union_area = box1_area + box2_area - inter_area

        #         return inter_area / union_area if union_area > 0 else 0
            
        #     def is_inside(box1, box2):
        #         """Checks if box1 is completely inside box2."""
        #         x1, y1, x2, y2 = box1
        #         x1_, y1_, x2_, y2_ = box2

        #         return x1_ <= x1 and y1_ <= y1 and x2_ >= x2 and y2_ >= y2

        #     merged = []
        #     while objects:
        #         label, base_box = objects.pop(0)
        #         x1, y1, x2, y2 = base_box
        #         merged_group = [(label, base_box)]

        #         # Find all overlapping or inside boxes
        #         i = 0
        #         while i < len(objects):
        #             other_label, other_box = objects[i]
        #             if compute_iou(base_box, other_box) > iou_threshold or is_inside(other_box, base_box) or is_inside(base_box, other_box):
        #                 merged_group.append((other_label, other_box))
        #                 objects.pop(i)  # Remove and reprocess
        #             else:
        #                 i += 1

        #         # Merge all grouped bounding boxes into one
        #         merged_x1 = min(box[1][0] for box in merged_group)
        #         merged_y1 = min(box[1][1] for box in merged_group)
        #         merged_x2 = max(box[1][2] for box in merged_group)
        #         merged_y2 = max(box[1][3] for box in merged_group)

        #         merged.append((label, (merged_x1, merged_y1, merged_x2, merged_y2)))

        #     return merged

        def merge_bounding_boxes(boxes):
            
            def do_boxes_overlap(box1, box2):
                """Checks if two bounding boxes overlap (including touching)."""
                x1, y1, x2, y2 = box1
                x1_, y1_, x2_, y2_ = box2
                return not (x2 < x1_ or x2_ < x1 or y2 < y1_ or y2_ < y1)

            # Step 1: Ensure all boxes are in (x1, y1, x2, y2) format
            labels = [label for label, box in boxes if len(box) == 4]
            boxes = [box for label, box in boxes if len(box) == 4]
            # boxes = [box for box in boxes if box is not None]  # Remove invalid entries

            # Step 2: Build adjacency list (graph of overlapping bounding boxes)
            adjacency_list = {i: set() for i in range(len(boxes))}
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if do_boxes_overlap(boxes[i], boxes[j]):
                        adjacency_list[i].add(j)
                        adjacency_list[j].add(i)

            # Step 3: Find connected components (clusters of overlapping boxes)
            visited = set()
            merged_boxes = []

            def dfs(node, cluster):
                """Depth-First Search (DFS) to find all connected bounding boxes."""
                if node in visited:
                    return
                visited.add(node)
                cluster.append(boxes[node])
                for neighbor in adjacency_list[node]:
                    dfs(neighbor, cluster)

            for i in range(len(boxes)):
                if i not in visited:
                    cluster = []
                    dfs(i, cluster)

                    # Step 4: Merge the bounding boxes in the found cluster
                    merged_x1 = min(b[0] for b in cluster)
                    merged_y1 = min(b[1] for b in cluster)
                    merged_x2 = max(b[2] for b in cluster)
                    merged_y2 = max(b[3] for b in cluster)

                    merged_boxes.append((labels[i], (merged_x1, merged_y1, merged_x2, merged_y2)))

            return merged_boxes
        
        # Merge overlapping new objects
        merged_new_objects = merge_bounding_boxes(new)

        for label, bbox in merged_new_objects:
            x1, y1, x2, y2 = bbox
            if x2 - x1 < 200 and y2 - y1 < 200:
                continue

            image_list.append(dirty_image_draw[y1:y2, x1:x2])
            label_list.append(label)
            # cv2.imwrite(f'objects/{label}.jpg', dirty_image_draw[y1:y2, x1:x2])
            # cv2.rectangle(dirty_image_draw, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red for new objects
            # cv2.putText(dirty_image_draw, f"New: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # cv2.imwrite('result.jpg', dirty_image_draw)
        # plt.figure(figsize=(12, 6))
        # plt.imshow(cv2.cvtColor(dirty_image_draw, cv2.COLOR_BGR2RGB))
        # plt.axis("off")
        # plt.title("Detected Changes with Merged Objects")
        # plt.show()
    
        return (image_list, label_list)


    def process(self):
        """Executes the full pipeline: warp, segment, detect, and compare."""
        print("Warping dirty image to clean image...")
        self.warp_dirty_to_clean()
        print("Applying SAM segmentation and YOLO detection...")
        displaced, removed, new = self.segment_and_detect()
        return self.draw_results(displaced, removed, new)    
    