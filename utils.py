import os
import glob
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms.functional import resize as F_resize
from torchvision.transforms import InterpolationMode
from osgeo import gdal


def compute_metrics(label_true, label_pred, num_classes=6,
                    compute_weighted=True, return_per_class=True):

    y_true = label_true.ravel()
    y_pred = label_pred.ravel()

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            confusion[t, p] += 1

    iou_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    class_metrics = {} if return_per_class else None

    for cls in range(num_classes):
        tp = confusion[cls, cls]
        fp = confusion[:, cls].sum() - tp
        fn = confusion[cls, :].sum() - tp

        union = tp + fp + fn
        iou = tp / (union + 1e-10)
        iou_list.append(iou)

        precision = tp / (tp + fp + 1e-10)
        precision_list.append(precision)

        recall = tp / (tp + fn + 1e-10)
        recall_list.append(recall)

        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        f1_list.append(f1)

        if return_per_class:
            class_metrics[f"Class_{cls}"] = {
                "IoU": iou, "F1_Score": f1,
                "Precision": precision, "Recall": recall
            }

    mIoU = np.mean(iou_list) if iou_list else 0.0
    macro_f1 = np.mean(f1_list) if f1_list else 0.0
    m_precision = np.mean(precision_list) if precision_list else 0.0
    m_recall = np.mean(recall_list) if recall_list else 0.0

    correct = np.trace(confusion)
    total = np.sum(confusion)
    oa = correct / (total + 1e-10)

    if compute_weighted:
        class_counts = [confusion[i, :].sum() for i in range(1, num_classes)]
        total_samples = sum(class_counts)
        if total_samples > 0:
            weights = [c / total_samples for c in class_counts]
            weighted_f1 = sum(f1_list[i] * weights[i-1] for i in range(1, num_classes))
            weighted_iou = sum(iou_list[i] * weights[i-1] for i in range(1, num_classes))
        else:
            weighted_f1 = weighted_iou = 0.0
    else:
        weighted_f1 = weighted_iou = None

    binary_confusion = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        true_bin = 0 if t == 0 else 1
        pred_bin = 0 if p == 0 else 1
        binary_confusion[true_bin, pred_bin] += 1

    tp_bin = binary_confusion[1, 1]
    fp_bin = binary_confusion[0, 1]
    fn_bin = binary_confusion[1, 0]

    union_bin = tp_bin + fp_bin + fn_bin
    iou_bin = tp_bin / (union_bin + 1e-10)
    precision_bin = tp_bin / (tp_bin + fp_bin + 1e-10)
    recall_bin = tp_bin / (tp_bin + fn_bin + 1e-10)
    f1_bin = 2 * precision_bin * recall_bin / (precision_bin + recall_bin + 1e-10)

    f1_class1 = f1_list[1] if num_classes > 1 else 0.0

    results = {
        "mIoU": mIoU,
        "Macro-average F1": macro_f1,
        "MPrecision": m_precision,
        "MRecall": m_recall,
        "Overall Accuracy": oa,
        "Binary IoU": iou_bin,
        "Binary F1 Score": f1_bin,
        "Binary Precision": precision_bin,
        "Binary Recall": recall_bin,
        "F1_1": f1_class1,
        "MF1": f1_class1
    }

    if compute_weighted:
        results["Frequency Weighted F1 (Excluding Background)"] = weighted_f1
        results["Frequency Weighted IoU (Excluding Background)"] = weighted_iou

    if return_per_class:
        results["Per-Class Metrics"] = class_metrics

    return results


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, extension_img="tif", extension_lab="tif",
                 image_size=None, return_file_dir=False, skip_zero_label=False):
        
        self.images_dir = self._find_files_by_extension(image_dir, extension_img)
        self.labels_dir = self._find_files_by_extension(mask_dir, extension_lab)

        self.valid_files = []
        for img_path in self.images_dir:
            base_name = os.path.basename(img_path)
            label_path = os.path.join(mask_dir, base_name)
            if label_path in self.labels_dir:
                # Extract class label from filename: e.g., "5_xxx.tif" -> 5
                cls_label = int(base_name.split('_')[0])
                if skip_zero_label and cls_label == 0:
                    continue
                self.valid_files.append((img_path, label_path))

        print(f"Found {len(self.valid_files)} valid samples (skip_zero_label={skip_zero_label}).")
        self.image_size = image_size
        self.return_file_dir = return_file_dir

    @staticmethod
    def _find_files_by_extension(directory, extension):
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(extension)]

    def _read_multiband_images(self, image_path):
        ds = gdal.Open(image_path)
        bands = [ds.GetRasterBand(i).ReadAsArray() for i in range(1, ds.RasterCount + 1)]
        return np.stack(bands, axis=0)

    def _read_singleband_labels(self, label_path):
        ds = gdal.Open(label_path)
        return ds.GetRasterBand(1).ReadAsArray()

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        img_path, lbl_path = self.valid_files[idx]
        image = self._read_multiband_images(img_path)
        label = self._read_singleband_labels(lbl_path)
        cls_label = int(os.path.basename(img_path).split('_')[0])

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        if self.image_size is not None:
            image = F_resize(image.unsqueeze(0), self.image_size, interpolation=InterpolationMode.NEAREST).squeeze(0)
            label = F_resize(label.unsqueeze(0).unsqueeze(0).float(),
                             self.image_size,
                             interpolation=InterpolationMode.NEAREST).squeeze().long()

        if self.return_file_dir:
            return {"image": image, "cls_label": torch.tensor(cls_label, dtype=torch.long),
                    "mask": label, "file_path": img_path}
        else:
            return {"image": image, "cls_label": torch.tensor(cls_label, dtype=torch.long), "mask": label}

class RSDataset(Dataset):
    def __init__(self, images_dir, labels_dir, extension_img="tif", extension_lab="tif",
                 image_size=None, return_file_dir=False):
        self.images_dir = find_files_by_extension(images_dir, extension_img)
        self.labels_dir = find_files_by_extension(labels_dir, extension_lab)
        print(f"Found {len(self.images_dir)} images.")
        self.image_size = image_size
        self.return_file_dir = return_file_dir

    def read_multiband_images(self, image_path):
        dataset = gdal.Open(image_path)
        num_bands = dataset.RasterCount
        bands = [dataset.GetRasterBand(i).ReadAsArray() for i in range(1, num_bands + 1)]
        return np.stack(bands, axis=0)

    def read_singleband_labels(self, label_path):
        dataset = gdal.Open(label_path)
        return dataset.GetRasterBand(1).ReadAsArray()

    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):
        image = self.read_multiband_images(self.images_dir[idx])
        label = self.read_singleband_labels(self.labels_dir[idx])

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()

        if self.image_size is not None:
            image = F.interpolate(image.unsqueeze(0), size=self.image_size, mode='nearest').squeeze(0)
            label = F.interpolate(label.unsqueeze(0).unsqueeze(0).float(), size=self.image_size, mode='nearest').squeeze().long()

        if self.return_file_dir:
            return image, label, self.images_dir[idx]
        else:
            return image, label


def init_weights(m):
    for m in m.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            

def find_files_by_extension(folder_path, extension="tif"):
    search_pattern = os.path.join(folder_path, f"*.{extension}")
    return glob.glob(search_pattern)

def run_epoch_Classification_Network(loader, model, criterion, optimizer, is_train=True, device='cuda'):
    total_loss = 0.0
    model.train() if is_train else model.eval()

    label_true = torch.LongTensor().to(device)
    label_pred = torch.LongTensor().to(device)

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            images = batch["image"].to(device)
            cls_labels = batch["mask"].to(device)

            outputs = model(images)

            loss = sum(criterion(output, cls_labels) for output in outputs) / len(outputs)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.cpu().item()

            label_true = torch.cat((label_true, cls_labels.data), dim=0)

            avg_output = torch.stack(outputs, dim=0).mean(dim=0)
            label_pred = torch.cat((label_pred, avg_output.argmax(dim=1).data), dim=0)

    total_loss /= len(loader)
    return total_loss, label_true, label_pred

def run_epoch_Segmentation_Network(loader, model, criterion, optimizer, is_train=True, device='cuda'):
    total_loss = 0.0
    model.train() if is_train else model.eval()

    label_true = torch.LongTensor().to(device)
    label_pred = torch.LongTensor().to(device)

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            images = batch["image"].to(device)
            seg_labels = batch["mask"].to(device)

            binary_labels = (seg_labels > 0).long()

            outputs = model(images)

            loss = sum(criterion(output, binary_labels) for output in outputs) / len(outputs)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.cpu().item()

            label_true = torch.cat((label_true, seg_labels.data), dim=0)

            avg_output = torch.stack(outputs, dim=0).mean(dim=0) 
            label_pred = torch.cat((label_pred, avg_output.argmax(dim=1).data), dim=0)

    total_loss /= len(loader)
    return total_loss, label_true, label_pred


def run_epoch_Dual_Task(loader, model, criterion, optimizer, is_train=True, device='cuda'):
    total_loss = 0.0
    model.train() if is_train else model.eval()

    label_true = torch.LongTensor().to(device)
    label_pred = torch.LongTensor().to(device)

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            images = batch["image"].to(device)
            seg_labels = batch["mask"].to(device)
            
            outputs = model(images)

            loss = sum(criterion(output, seg_labels) for output in outputs) / len(outputs)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.cpu().item()

            label_true = torch.cat((label_true, seg_labels.data), dim=0)

            avg_output = torch.stack(outputs, dim=0).mean(dim=0) 
            label_pred = torch.cat((label_pred, avg_output.argmax(dim=1).data), dim=0)

    total_loss /= len(loader)
    return total_loss, label_true, label_pred

def run_epoch_DHD_Net(loader, model, criterion, optimizer=None, is_train=True, device='cuda'):
    total_loss = 0.0
    model.train() if is_train else model.eval()
    label_true = torch.LongTensor().to(device)
    label_pred = torch.LongTensor().to(device)

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = (criterion(outputs[0], labels) + criterion(outputs[1], labels) +
                    criterion(outputs[2], labels) + criterion(outputs[3], labels)) / 4

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.cpu().item()
            label_true = torch.cat((label_true, labels.data), dim=0)
            label_pred = torch.cat((label_pred,((outputs[0]+outputs[1]+outputs[2]+outputs[3])/4).argmax(dim=1).data), dim=0)

    total_loss /= len(loader)
    return total_loss, label_true, label_pred

def run_epoch(loader, model, criterion, optimizer=None, is_train=True, device='cuda'):
    total_loss = 0.0
    model.train() if is_train else model.eval()
    label_true = torch.LongTensor().to(device)
    label_pred = torch.LongTensor().to(device)

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.cpu().item()
            label_true = torch.cat((label_true, labels.data), dim=0)
            label_pred = torch.cat((label_pred,outputs.argmax(dim=1).data), dim=0)

    total_loss /= len(loader)
    return total_loss, label_true, label_pred