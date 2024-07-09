import os
import time
import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO

output_folder = 'Dedublicated_images'  # Выходная корневая папка

# Полигоны для каждой камеры
polygons = {
    247: np.array([[1273, 1410], [2565, 694], [2385, 602], [2077, 282], [501, 794], [905, 1058], [1185, 1282]]),
    248: np.array([[1061, 642], [2305, 734], [2681, 1154], [2681, 1250], [245, 1222]]),
    249: np.array([[325, 618], [1953, 1222], [2201, 630], [2213, 378], [693, 218], [489, 466]]),
    252: np.array([[485, 914], [1549, 1510], [2105, 1506], [2273, 1266], [2573, 990], [1129, 610], [921, 750]])
}

simbol_dict = {0: 'plate', 1: 'А', 2: 'В', 3: 'Е', 4: 'К', 5: 'М', 6: 'Н', 7: 'О', 8: 'Р', 9: 'С',
               10: 'Т', 11: 'У', 12: 'Х',
               13: '0', 14: '1', 15: '2', 16: '4', 17: '5', 18: '6',
               19: '7', 20: '8', 21: '9', 22: '3', 23: 'RUS', 24: '?', 25: '??', 26: '???'}


def count_files(folder_path, formats):
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(formats):
                count += 1
    return count


class ImageDedublicator:

    def __init__(self,
                 source_dir,
                 camera_id=248):

        self.frame = None

        self.source_dir = source_dir
        self.root = None

        self.model_numberplate = YOLO("Models/numberplate_model_v3.pt")
        self.model_license_simbol = YOLO("Models/number_rus_100.pt")
        self.model_car = YOLO('Models/yolov8n.pt')

        self.detections_numberplate = None
        self.detections_license_simbol = None
        self.detections_car = None

        self.numberplate_frame = None
        self.car_frame = None

        self.camera_id = camera_id
        self.car_number = None

        self.polygon_zone = sv.PolygonZone(polygon=polygons.get(self.camera_id),
                                           triggering_anchors=(
                                               [sv.Position.BOTTOM_CENTER, sv.Position.BOTTOM_LEFT,
                                                sv.Position.BOTTOM_RIGHT]))

    def recognize_license_simbol(self):
        if self.detections_license_simbol.class_id.size > 0:
            sorted_indices = self.detections_license_simbol.xyxy[:, 0].argsort()  # индекс сортировки
            sorted_class_names = self.detections_license_simbol.data['class_name'][sorted_indices]  # имена классов в str
            detected_class_names = [int(string) for string in sorted_class_names]
            detected_classes = [simbol_dict[detected_class_name] for detected_class_name in detected_class_names]
            self.car_number = ''.join(detected_classes)  # Автомобильный номер
            return self.car_number
        else:
            self.car_number = 'Не распознано!'
            return self.car_number

    def predict_car(self):
        selected_classes = [2, 3, 7, 5]  # классы 'car', 'motorcycle', 'truck', 'bus'
        result_car = self.model_car(self.frame,
                                    conf=0.4,
                                    agnostic_nms=True,
                                    augment=True,
                                    verbose=False)[0]
        self.detections_car = sv.Detections.from_ultralytics(result_car)
        self.detections_car = self.detections_car[np.isin(self.detections_car.class_id, selected_classes)]
        mask_car = self.polygon_zone.trigger(self.detections_car)
        self.detections_car = self.detections_car[mask_car]

    def predict_numberplate(self):
        result_numberplate = self.model_numberplate(self.frame,
                                                    conf=0.1,
                                                    verbose=False)[0]
        self.detections_numberplate = sv.Detections.from_ultralytics(result_numberplate)
        mask_np = self.polygon_zone.trigger(self.detections_numberplate)
        self.detections_numberplate = self.detections_numberplate[mask_np]

    def predict_license_simbol(self):
        selected_classes_license_model = list(range(1, 23))  # выбранные классы, 0 - это рамка номера
        results_license_simbol = self.model_license_simbol(self.numberplate_frame,
                                                           conf=0.2,
                                                           max_det=20,
                                                           iou=0.7,
                                                           agnostic_nms=True,
                                                           augment=True,
                                                           verbose=False)[0]
        self.detections_license_simbol = sv.Detections.from_ultralytics(results_license_simbol)
        self.detections_license_simbol = self.detections_license_simbol[
            np.isin(self.detections_license_simbol.data['class_name'],
                    selected_classes_license_model)]
        self.recognize_license_simbol()
        return self.car_number

    def crop_car(self):
        if self.detections_car.class_id.size > 0:  # если обнаружены номера
            for detection_car in self.detections_car.xyxy:
                self.car_frame = sv.crop_image(image=self.frame, xyxy=detection_car)
                return self.car_frame
        else:
            self.car_frame = None
            return self.car_frame

    def crop_numberplate(self):
        if self.detections_numberplate.class_id.size > 0:  # если обнаружены номера
            for detection_numberplate in self.detections_numberplate.xyxy:
                self.numberplate_frame = sv.crop_image(image=self.frame, xyxy=detection_numberplate)
                return self.numberplate_frame
        else:
            self.numberplate_frame = None
            return self.numberplate_frame

    def run(self):
        start_time = time.time()  # Текущее время перед началом выполнения кода

        FORMATS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
        num_images = count_files(self.source_dir, FORMATS)
        print(f"КОЛИЧЕСТВО ФАЙЛОВ (в том числе во вложенных папках): {num_images}")
        print('-' * 100)

        index_frame = 0  # Число обработанных кадров
        saved_frame = 0  # Число сохранённых кадров
        numberplate_list = ['Не распознано!']  # Стартовый список распознанных номеров

        for root, dirs, files in os.walk(self.source_dir):
            relative_path = os.path.relpath(root, self.source_dir)
            output_path = os.path.join(output_folder, relative_path)
            os.makedirs(output_path, exist_ok=True)  # Создаем выходную папку

            for file in files:
                if file.endswith(FORMATS):
                    index_frame += 1
                    video_source_name = file
                    video_path = os.path.join(root, video_source_name)  # Получаем путь к текущему файлу

                    print(f'{index_frame}. Взят на обработку файл {video_path}')

                    self.frame = cv2.imread(video_path)

                    self.predict_numberplate()  # обнаруживаем рамку номера
                    self.crop_numberplate()  # вырезаем рамку номера
                    self.predict_license_simbol()  # обнаруживаем символы на номере

                    print(f'Распознан номер: {self.car_number}')

                    if self.car_number not in numberplate_list and len(self.car_number) > 2:
                        saved_frame += 1
                        print('Это новый автомобиль. Сохраняем изображение!')
                        self.predict_car()  # обнаруживаем авто
                        self.crop_car()  # вырезаем авто

                        output_filename = os.path.join(output_path, video_source_name)
                        cv2.imwrite(output_filename, self.car_frame)  # Сохранение обработанного изображения

                    else:
                        print('Такой автомобиль уже существует. Пропускаем изображение...')

                    if self.car_number not in numberplate_list:
                        numberplate_list.append(self.car_number)  # добавляем новый номер в список

                    print('-' * 30)

        end_time = time.time()  # Запоминаем текущее время после выполнения кода
        print('-' * 70)
        print(f'Время обработки всех изображений: {round(end_time - start_time)} сек')
        print(f'Средняя скорость обработки: {(num_images/(end_time - start_time)):.1f} кадров/сек')
        print(f'Кол-во исходных изображений: {num_images}')
        print(f'Кол-во отобранных изображений: {saved_frame}')
        print(f'Отобрано: {((saved_frame/num_images)*100):.1f} %')
        print('-' * 70)

        print('РАБОТА ПРОГРАММЫ ЗАВЕРШЕНА!!!')
