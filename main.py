from core import ImageDedublicator


source_folder = 'Source_images'  # исходные изображения

process = ImageDedublicator(source_dir=source_folder,
                            camera_id=248)  # выберите камеру 247, 248, 249, 252

process.run()
