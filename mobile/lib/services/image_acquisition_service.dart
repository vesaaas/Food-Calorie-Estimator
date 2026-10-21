import 'package:image_picker/image_picker.dart';

abstract interface class ImageAcquisitionService {
  Future<XFile?> takePhoto();

  Future<XFile?> chooseFromGallery();
}

class ImagePickerAcquisitionService implements ImageAcquisitionService {
  ImagePickerAcquisitionService({ImagePicker? picker})
    : _picker = picker ?? ImagePicker();

  final ImagePicker _picker;

  @override
  Future<XFile?> takePhoto() => _picker.pickImage(source: ImageSource.camera);

  @override
  Future<XFile?> chooseFromGallery() =>
      _picker.pickImage(source: ImageSource.gallery);
}
