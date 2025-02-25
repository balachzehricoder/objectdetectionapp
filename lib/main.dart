import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter_plus/tflite_flutter_plus.dart' as tfl;
import 'package:tflite_flutter_helper_plus/tflite_flutter_helper_plus.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'dart:typed_data';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(camera: cameras.first));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;
  const MyApp({Key? key, required this.camera}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: ObjectDetectionScreen(camera: camera),
    );
  }
}

class ObjectDetectionScreen extends StatefulWidget {
  final CameraDescription camera;
  const ObjectDetectionScreen({Key? key, required this.camera}) : super(key: key);

  @override
  _ObjectDetectionScreenState createState() => _ObjectDetectionScreenState();
}

class _ObjectDetectionScreenState extends State<ObjectDetectionScreen> {
  late CameraController _cameraController;
  late tfl.Interpreter _interpreter;
  bool _isDetecting = false;
  List<Map<String, dynamic>> _results = [];
  List<String> _labels = [];

  @override
  void initState() {
    super.initState();
    _initCamera();
    _loadModel();
  }

  Future<void> _initCamera() async {
    _cameraController = CameraController(widget.camera, ResolutionPreset.medium);
    await _cameraController.initialize();
    if (!mounted) return;
    setState(() {});
    _startDetection();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await tfl.Interpreter.fromAsset("assets/ssd_mobilenet.tflite");
      String labelString = await rootBundle.loadString("assets/labels.txt");
      _labels = labelString.split("\n").map((e) => e.trim()).toList();
      setState(() {});
      print("✅ Model & Labels Loaded Successfully!");
    } catch (e) {
      print("❌ Failed to Load Model: $e");
    }
  }

  void _startDetection() {
    _cameraController.startImageStream((CameraImage image) async {
      if (_isDetecting) return;
      _isDetecting = true;

      var inputImage = _preProcessImage(image);
      var outputBuffer = List.generate(1, (index) => List.filled(10, 0.0));

      _interpreter.run(inputImage.buffer.asUint8List(), outputBuffer);
      var results = _postProcessOutput(outputBuffer);

      setState(() {
        _results = results;
      });

      _isDetecting = false;
    });
  }

TensorImage _preProcessImage(CameraImage image) {
  var inputShape = _interpreter.getInputTensor(0).shape;
  int inputSize = inputShape[1];

  img.Image convertedImage = _convertCameraImage(image);
  convertedImage = img.copyResize(convertedImage, width: inputSize, height: inputSize);

  TensorImage tensorImage = TensorImage(tfl.TfLiteType.float32);
  tensorImage.loadImage(convertedImage);

  ImageProcessor imageProcessor = ImageProcessorBuilder()
    .add(ResizeOp(inputSize, inputSize, ResizeMethod.bilinear)) // Correct method
    .add(NormalizeOp(127.5, 127.5))
    .build();


  return imageProcessor.process(tensorImage);
}


  img.Image _convertCameraImage(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final img.Image imgBuffer = img.Image(width, height);

    final int uvRowStride = image.planes[1].bytesPerRow;
    final int uvPixelStride = image.planes[1].bytesPerPixel!;

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final int uvIndex = uvRowStride * (y ~/ 2) + (x ~/ 2) * uvPixelStride;
        final int index = y * width + x;

        final int yValue = image.planes[0].bytes[index];
        final int uValue = image.planes[1].bytes[uvIndex];
        final int vValue = image.planes[2].bytes[uvIndex];

        imgBuffer.setPixel(x, y, img.getColor(yValue, uValue, vValue));
      }
    }
    return imgBuffer;
  }

  List<Map<String, dynamic>> _postProcessOutput(List<List<double>> output) {
    List<Map<String, dynamic>> results = [];
    for (int i = 0; i < output[0].length && i < _labels.length; i++) {
      double confidence = output[0][i];
      if (confidence > 0.4) {
        results.add({
          "label": _labels[i],
          "confidence": confidence,
        });
      }
    }
    return results;
  }

  @override
  void dispose() {
    _cameraController.dispose();
    _interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_cameraController.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    return Scaffold(
      appBar: AppBar(title: const Text('Object Detection')),
      body: Stack(
        children: [
          CameraPreview(_cameraController),
          if (_results.isNotEmpty)
            Positioned(
              bottom: 20,
              left: 10,
              child: Column(
                children: _results.map((result) {
                  return Text(
                    "${result['label']} - ${(result['confidence'] * 100).toStringAsFixed(0)}%",
                    style: const TextStyle(color: Colors.white, fontSize: 18),
                  );
                }).toList(),
              ),
            ),
        ],
      ),
    );
  }
}
