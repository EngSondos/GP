import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:csv/csv.dart';
import 'dart:async' show Future;
import 'package:flutter/services.dart' show rootBundle;

List<List<dynamic>> data = [];
String text = "";

class TableLayout extends StatefulWidget {
  @override
  TableLayoutState createState() => new TableLayoutState();
}

class TableLayoutState extends State<TableLayout> {
  loadAsset(table) async {
    List<List<dynamic>> csvTable = [];
    csvTable = CsvToListConverter().convert(table);
    data = csvTable;
    print(data[1][0]);
    text = diagnosis(data[1][0], data[1][1], data[1][2], data[1][3]);
    // setState(() {});

    // print("$data sondos");
  }

  String diagnosis(Eosinophil, Lymphocyte, Monocyte, Neutrophil) {
    text = " ";
    int sum = Eosinophil + Lymphocyte + Monocyte + Neutrophil;
    double PercentOfEosinophil = Eosinophil / sum * 100;
    double PercentOfLymphocyte = Lymphocyte / sum * 100;
    double PercentOfMonocyte = Monocyte / sum * 100;
    double PercentOfNeutrophil = Neutrophil / sum * 100;
    print(
        "$PercentOfNeutrophil, $PercentOfMonocyte , $PercentOfLymphocyte ,$PercentOfEosinophil");
    if (PercentOfLymphocyte < 20.0) {
      text +=
          "\n \n decrease in Lymphocytes may be caused by : chemotherapy ,HIV infection ,leukemia , sepsis ,radiation exposure, either accidental or from radiation therapy ";
    } else if (PercentOfLymphocyte > 40.0) {
      text +=
          "\n \n  increase in Lymphocytes may be caused by :chronic infection ,mononucleosis , leukemia ,viral infection, such as the mumps or measles";
    }
    if (PercentOfNeutrophil < 40.0) {
      text +=
          "\n \n decrease in Neutrophil may be caused by :anemia , bacterial infection , chemotherapy ,influenza or other viral illnesses ,radiation exposure";
    } else if (PercentOfNeutrophil > 60.0) {
      text +=
          "\n \n increase in Neutrophil may be caused by :infection , gout , rheumatoid arthritis , thyroiditis , trauma";
    }
    if (PercentOfEosinophil > 3.0) {
      text +=
          "\n \n increase in Eosinophil may be caused by : an allergic reaction , parasitic infection ";
    }
    if (PercentOfMonocyte < 4.0) {
      text +=
          "\n \n  decrease in Monocyte may be caused by :bloodstream infection ,chemotherapy , bone marrow disorder ,skin infections";
    } else if (PercentOfMonocyte > 8.0) {
      text +=
          "\n \n increase in Monocyte may be caused by : chronic inflammatory disease , tuberculosis , viral infection, such as measles, mononucleosis, and mumps ";
    }

    return (text);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        backgroundColor: Colors.grey[800],
        appBar: AppBar(
          title: Text("Result",
              textAlign: TextAlign.center,
              style: TextStyle(
                color: Colors.grey[300],
              )),
          backgroundColor: Colors.grey[800],
        ),
        body: Container(
            margin: EdgeInsets.only(top: 40.0, bottom: 40.0),
            child: Column(
              children: [
                Container(
                  margin: EdgeInsets.only(bottom: 10.0),
                  child: Text(
                    "WBC Count:",
                    style: TextStyle(color: Colors.grey[500], fontSize: 15.0),
                  ),
                ),
                Table(
                  border: TableBorder.all(
                      width: 2.0, color: Color.fromARGB(255, 59, 55, 55)),
                  children: data.map((item) {
                    return TableRow(
                        children: item.map((row) {
                      return Container(
                        child: Padding(
                          padding: const EdgeInsets.all(8.0),
                          child: Text(
                            row.toString(),
                            style: TextStyle(
                                color: Colors.grey[500],
                                fontSize: 12.0,
                                fontWeight: FontWeight.bold),
                          ),
                        ),
                      );
                    }).toList());
                  }).toList(),
                ),
                Container(
                    margin: EdgeInsets.only(top: 20, left: 10, right: 10),
                    child: Text(text,
                        style: TextStyle(
                          color: Colors.grey[500],
                        ))),
              ],
            )));
  }
}
