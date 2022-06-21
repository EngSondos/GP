// import 'dart:html';
import 'dart:io';
// import 'dart:typed_data';
import 'package:http/http.dart' as http;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter/services.dart';
import 'readcsv.dart';
// import 'dart:convert';
import 'package:flutter_overlay_loader/flutter_overlay_loader.dart';

void main() {
  runApp(
    MaterialApp(
      debugShowCheckedModeBanner: false,
      home: MyApp(),
    ),
  );
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Medical d',
      debugShowCheckedModeBanner: false,
      home: MyHomePage(title: 'Medical'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  // int check = 0;
  ImagePicker _picker = ImagePicker();
  List<XFile>? images = [];
  late bool _isButtonDisabled;
  String message = "";

  Future<String> uploadimage() async {
    int check = 0;
    File selectedImages;
    final request = http.MultipartRequest(
        "POST",
        Uri.parse(
            "https://e4f4-156-208-149-26.eu.ngrok.io/upload")); //ngrok link +/upload
    final headers = {"Content-type": "multipart/form-data"};
    // selectedImages![0] = images![0];
    for (int i = 0; i < images!.length; i++) {
      selectedImages = File(images![i].path);
      print(selectedImages.path);
      request.files.add(http.MultipartFile("image",
          selectedImages.readAsBytes().asStream(), selectedImages.lengthSync(),
          filename: selectedImages.path.split("/").last));
      request.headers.addAll(headers);
    }
    final response = await request.send();

    http.Response res = await http.Response.fromStream(response);
    print(res.body);
    print(check);

    new TableLayoutState().loadAsset(res.body);
    // while (response.statusCode != 200) {}
    // check = 1;

    setState(() {});

    return Future.value("1");
  }

  Future pickImage() async {
    // final image = await ImagePicker().;
    final List<XFile>? selectedImages = await _picker.pickMultiImage();
    print(selectedImages?.length);
    if (selectedImages?.length == null) {
      return;
    } else {
      images!.addAll(selectedImages!);
    }

    // if (image == null) return;

    // final imageTemp = File(image.path);

    // setState(() => this.images = imageTemp);
    setState(() {});
  }

  Future pickImageC() async {
    try {
      final image = await ImagePicker().pickImage(source: ImageSource.camera);

      if (image == null) return;

      // final imageTemp = File(image.path);

      // setState(() => this.image = imageTemp);
      setState(() {});
    } on PlatformException catch (e) {
      print('Failed to pick image: $e');
    }
  }

  // File _file = File("zz");
  // Uint8List webImage = Uint8List(10);

  @override
  Widget build(BuildContext context) {
    _isButtonDisabled = false;
    if (images!.isEmpty) [_isButtonDisabled = true];
    return Scaffold(
      backgroundColor: Colors.grey[800],

      appBar: AppBar(
        title: Text("Medical diagnosis",
            textAlign: TextAlign.center,
            style: TextStyle(
              color: Colors.grey[300],
            )),
        backgroundColor: Colors.grey[800],
      ),
      // drawer: Drawer(),
      body: Center(
          child: Column(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          SizedBox(
            height: 400.0,
            child: images!.isNotEmpty
                ? Padding(
                    padding: EdgeInsets.all(5.0),
                    child: GridView.builder(
                        itemCount: images!.length,
                        gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                            crossAxisCount: 3),
                        itemBuilder: (BuildContext context, int index) {
                          return Padding(
                              padding: EdgeInsets.all(3.0),
                              child: Stack(
                                fit: StackFit.expand,
                                children: [
                                  Image.file(
                                    File(images![index].path),
                                    fit: BoxFit.cover,
                                  ),
                                  Positioned(
                                    top: 8,
                                    right: 4,
                                    child: Container(
                                      // color: Color.fromARGB(255, 136, 134, 134),
                                      child: IconButton(
                                        onPressed: () {
                                          images!.removeAt(index);
                                          setState(() {});
                                        },
                                        icon: Icon(Icons.delete),
                                        color:
                                            Color.fromARGB(255, 255, 255, 255),
                                      ),
                                    ),
                                  ),
                                ],
                              ));
                        }),
                  )
                : Text(
                    "No image selected",
                    style: TextStyle(color: Colors.grey[300], fontSize: 20.0),
                  ),
          ),
          Container(
            margin: EdgeInsets.only(top: 30.0),
            child: Row(
              children: [
                Expanded(
                    child: Container(
                  decoration: BoxDecoration(
                      border: Border.all(
                          color: Colors.transparent,
                          width: 10.0,
                          style: BorderStyle.solid),
                      borderRadius: BorderRadius.all(Radius.circular(10.0))),
                  margin: EdgeInsets.symmetric(horizontal: 20.0),
                  child: SizedBox.fromSize(
                    size: Size(20, 100), // button width and height
                    child: ClipRRect(
                      child: Material(
                        color: Colors.grey[800], // button color
                        child: InkWell(
                          splashColor: Colors.grey[800], // splash color
                          onTap: () {
                            pickImage();
                          }, // button pressed
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: <Widget>[
                              Icon(
                                Icons.image,
                                color: Colors.grey[300],
                                size: 40.0,
                              ), // icon
                              Text(
                                "Gallery",
                                style: TextStyle(
                                    color: Colors.grey[300], fontSize: 20.0),
                              ), // text
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                )
                    // decoration:
                    // BoxDecoration(
                    //     border: Border.all(color: Colors.grey ,width: 10.0,style: BorderStyle.solid)
                    //     ,borderRadius: BorderRadius.all(Radius.circular(10.0))),
                    // margin: EdgeInsets.symmetric(horizontal: 100.0),

                    ),
                // SizedBox(height: 100),
                // SizedBox(width: 100, height: 100,),
                Expanded(
                    child: Container(
                  decoration: BoxDecoration(
                      border: Border.all(
                          color: Colors.transparent,
                          width: 10.0,
                          style: BorderStyle.solid),
                      borderRadius: BorderRadius.all(Radius.circular(10.0))),
                  margin: EdgeInsets.symmetric(horizontal: 20.0),
                  child: SizedBox.fromSize(
                    size: Size(20, 100), // button width and height
                    child: ClipRRect(
                      child: Material(
                        color: Colors.grey[800], // button color
                        child: InkWell(
                          splashColor: Colors.grey[800], // splash color
                          onTap: () {
                            pickImageC();
                          }, // button pressed
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: <Widget>[
                              Icon(
                                Icons.photo_camera,
                                color: Colors.grey[300],
                                size: 40.0,
                              ),
                              // icon
                              Text("Take Photo",
                                  style: TextStyle(
                                      color: Colors.grey[300], fontSize: 20.0)),
                              // text
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                ))
              ],
            ),
          ),
          Center(
            child: TextButton(
              style: ButtonStyle(),
              onPressed: () async {
                String checker = "0";
                if (!_isButtonDisabled) {
                  Loader.show(
                    context,
                    progressIndicator: CircularProgressIndicator(
                        backgroundColor: Color.fromARGB(255, 15, 34, 127)),
                    overlayColor: Colors.black87,
                  );
                  Future<String> intFuture = uploadimage();
                  checker = await intFuture;

                  // checker = uploadimage();
                  // while (checker == "0") {}
                  Loader.hide();
                  Navigator.push(context, MaterialPageRoute(builder: (context) {
                    return new TableLayout();
                  }));
                }
              },
              child: Text(
                'diagnose',
                style: TextStyle(color: Colors.blueGrey, fontSize: 25.0),
              ),
            ),
          ),
        ],
      )),
    );
  }
}
