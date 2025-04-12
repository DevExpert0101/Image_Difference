import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from lightglue import LightGlue, SIFT
from lightglue.utils import rbd
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from ultralytics import YOLO, FastSAM
import gc



class ImageProcessor:
    def __init__(self, sam_model_path = "models/sam_vit_h_4b8939.pth", yolo_model_path = "models/yolov10x.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.clean_image = None
        self.dirty_image = None
        # Initialize feature extractor and matcher
        self.extractor = SIFT()
        self.matcher = LightGlue(features="sift").eval().to(self.device)

        # Load SAM model
        # self.sam = sam_model_registry["vit_h"](checkpoint=sam_model_path).to(self.device)
        # self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.segmentor = FastSAM('models/FastSAM-x.pt').to(self.device)

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
        
        # del image0, image1, feats0, feats1, matches01, kpts0, kpts1, matches
        # torch.cuda.empty_cache()
        # gc.collect()
        
        return aligned_dirty_image

    
    def segment_and_detect(self):
        """Applies SAM for segmentation and YOLO for object detection on the aligned image."""
        clean_rgb = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2RGB)
        dirty_rgb = cv2.cvtColor(self.dirty_image, cv2.COLOR_BGR2RGB)
        
        # Generate segmentation masks
        with torch.no_grad():
            # clean_masks = self.mask_generator.generate(clean_rgb)
            # dirty_masks = self.mask_generator.generate(dirty_rgb)
            clean_masks = self.segmentor(clean_rgb, device='cuda', retina_masks=True, conf=0.4, iou=0.9)
            dirty_masks = self.segmentor(dirty_rgb, device='cuda', retina_masks=True, conf=0.4, iou=0.9)

        # Detect objects
            clean_objects = self.detect_objects_fastsam(clean_rgb, clean_masks[0], 'clean')
            dirty_objects = self.detect_objects_fastsam(dirty_rgb, dirty_masks[0], 'dirty')

        return self.compare_objects(clean_objects, dirty_objects)

    def detect_objects(self, image, masks, category):
        """Detects objects in segmented image regions using YOLO."""
        detected_objects = []
        for mask in masks:
            segmentation = mask['segmentation']
            x, y, w, h = cv2.boundingRect(segmentation.astype(np.uint8))
            cropped_object = image[y:y+h, x:x+w].copy()
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
    
    def detect_objects_fastsam(self, image, masks, category):
        """Detects objects in segmented image regions using YOLO."""
        detected_objects = []
        h, w, _ = image.shape
        output_image = image.copy()
        
        if hasattr(masks, 'boxes') and masks.boxes is not None:
            boxes = masks.boxes.xyxy.cpu().numpy()
            cls_indices = masks.boxes.cls.cpu().numpy()

            for box, cls in zip(boxes, cls_indices):
                x1, y1, x2, y2 = map(int, box)
                
                if x2 - x1 < 100 and y2 - y1 < 100:
                    continue
                if x2 -x1 > w * 0.7 or y2 - y1 > h * 0.6:
                    continue
                
                
                padding = 400                                
                cropped_object = image[max(0,y1 - padding) : min(h, y2 + padding), max(0, x1 - padding):min(w, x2 + padding)].copy()

                yolo_result = self.yolo_model(cropped_object, verbose=False)

                for r in yolo_result:
                    for box in r.boxes:
                        xx1, yy1, xx2, yy2 = map(int, box.xyxy[0])
                        conf = box.conf.item()
                        cls = int(box.cls.item())
                        label = self.yolo_model.names[cls]

                        # Map back to original image coordinates
                        xx1, xx2 = max(0, x1 - padding) + xx1, max(0, x1 - padding) + xx2
                        yy1, yy2 = max(0, y1 - padding) + yy1, max(0, y1 - padding) + yy2

                        detected_objects.append((label, (xx1, yy1, xx2, yy2)))        
                        
                        cv2.rectangle(output_image, (xx1, yy1), (xx2, yy2), (0, 255, 0), 2)
                        cv2.putText(output_image, f"{label} {conf:.2f}", (xx1, yy1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return detected_objects

    # def compare_objects(self, clean_objs, dirty_objs, threshold=200):
    #     """Compares objects between clean and dirty images to find new, displaced, or removed objects."""
    #     from scipy.spatial import distance
    #     h, w, _ = self.clean_image.shape
    #     displaced_objects, removed_objects, new_objects = [], [], []
    #     clean_dict = {label: [] for label, _ in clean_objs}
    #     for label, bbox in clean_objs:
    #         x1, y1, x2, y2 = bbox
    #         if x2 - x1 < 100 and y2 - y1 < 100:
    #             continue
    #         if x2 - x1 > w * 0.7 or y2 - y1 > h * 0.7:
    #             continue
    #         clean_dict[label].append(bbox)

    #     dirty_dict = {label: [] for label, _ in dirty_objs}
    #     for label, bbox in dirty_objs:
    #         x1, y1, x2, y2 = bbox
    #         if x2 - x1 < 100 and y2 - y1 < 100:
    #             continue
    #         if x2 - x1 > w * 0.7 or y2 - y1 > h * 0.7:
    #             continue
    #         dirty_dict[label].append(bbox)

    #     # Find displaced and removed objects
    #     for label, clean_bbox_list in clean_dict.items():
    #         for clean_bbox in clean_bbox_list:
    #             found, min_dist, closest_bbox = False, float("inf"), None

    #             if label in dirty_dict:
    #                 for dirty_bbox in dirty_dict[label]:
    #                     dist = distance.euclidean(
    #                         [(clean_bbox[0] + clean_bbox[2]) / 2, (clean_bbox[1] + clean_bbox[3]) / 2],
    #                         [(dirty_bbox[0] + dirty_bbox[2]) / 2, (dirty_bbox[1] + dirty_bbox[3]) / 2]
    #                     )
    #                     if dist < threshold and dist < min_dist:
    #                         min_dist, closest_bbox, found = dist, dirty_bbox, True

    #             if found and min_dist > 10:
    #                 displaced_objects.append((label, closest_bbox))
    #             elif not found:
    #                 removed_objects.append((label, clean_bbox))

    #     # Find new objects
    #     for label, dirty_bbox_list in dirty_dict.items():
    #         for dirty_bbox in dirty_bbox_list:
    #             found = any(
    #                 distance.euclidean(
    #                     [(clean_bbox[0] + clean_bbox[2]) / 2, (clean_bbox[1] + clean_bbox[3]) / 2],
    #                     [(dirty_bbox[0] + dirty_bbox[2]) / 2, (dirty_bbox[1] + dirty_bbox[3]) / 2]
    #                 ) < threshold
    #                 for clean_bbox in clean_dict.get(label, [])
    #             )
    #             if not found:
    #                 new_objects.append((label, dirty_bbox))

    #     return displaced_objects, removed_objects, new_objects

    def compare_objects(self, clean_objs, dirty_objs, threshold=200):
        """Compares objects between clean and dirty images to find new, displaced, or removed objects."""
        from scipy.spatial import distance
        h, w, _ = self.clean_image.shape
        displaced_objects, removed_objects, new_objects = [], [], []
        clean_dict = {label: [] for label, _ in clean_objs}
        for label, bbox in clean_objs:
            x1, y1, x2, y2 = bbox
            if x2 - x1 < 100 and y2 - y1 < 100:
                continue
            if x2 - x1 > w * 0.7 or y2 - y1 > h * 0.7:
                continue
            clean_dict[label].append(bbox)

        dirty_dict = {label: [] for label, _ in dirty_objs}
        for label, bbox in dirty_objs:
            x1, y1, x2, y2 = bbox
            if x2 - x1 < 100 and y2 - y1 < 100:
                continue
            if x2 - x1 > w * 0.7 or y2 - y1 > h * 0.7:
                continue
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
                    displaced_objects.append((label, closest_bbox))
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

    def draw_results(self, displaced, removed, new):
        """Draws the results on the dirty image with merged overlapping objects."""
        dirty_image_draw = self.dirty_image.copy()
        clean_image_draw = self.clean_image.copy()
        
        h, w, _ = dirty_image_draw.shape
        image_list = []
        label_list = []
        

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
        merged_diplaced_objects = merge_bounding_boxes(displaced)

        for label, bbox in merged_new_objects:
            x1, y1, x2, y2 = bbox
            if x2 - x1 <  100 and y2 - y1 < 100:
                continue
            if x2 - x1 > w * 0.7 or y2 - y1 > h * 0.7:
                continue
            
            # cv2.rectangle(dirty_image_draw, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red for new objects
            # cv2.putText(dirty_image_draw, f"New: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            image_list.append(dirty_image_draw[y1:y2, x1:x2])
            label_list.append(label)

        for label, dirty_bbox in merged_diplaced_objects:
            x1, y1, x2, y2 = dirty_bbox            
            if x2 - x1 < 100 and y2 - y1 < 100:
                continue
            if x2 -x1 > w * 0.7 or y2 - y1 > h * 0.7:
                continue
            
            # cv2.rectangle(dirty_image_draw, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue for displaced objects
            # cv2.putText(dirty_image_draw, f"Moved: {label}", (x1, y1 - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            image_list.append(dirty_image_draw[y1:y2, x1:x2])
            label_list.append(label)
        

        return (image_list, label_list)


    def process(self):
        """Executes the full pipeline: warp, segment, detect, and compare."""
        print("Warping dirty image to clean image...")
        
        self.warp_dirty_to_clean()
        print("Applying SAM segmentation and YOLO detection...")
        displaced, removed, new = self.segment_and_detect()        
        return self.draw_results(displaced, removed, new)    
    