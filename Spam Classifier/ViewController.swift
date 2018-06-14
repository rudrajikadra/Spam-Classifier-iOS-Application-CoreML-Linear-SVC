//
//  ViewController.swift
//  Spam Classifier
//
//  Created by Rudra Jikadra on 12/02/18.
//  Copyright © 2018 RedRudy. All rights reserved.
//

import UIKit
import CoreML

class ViewController: UIViewController {

    @IBOutlet weak var outputLabel: UILabel!
    @IBOutlet weak var message_field: UITextField!
    @IBOutlet weak var checkBut: UIButton!
    
    
    override var prefersStatusBarHidden: Bool{
        return true
    }
    
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
        
        outputLabel.text = "Spam or Ham?"
        
    }
    
    
    
    
    @IBAction func check(_ sender: Any) {
        
        let message = message_field.text
        
        if message == "" {
            outputLabel.text = "Please Enter a valid Statement!"
        } else {
            outputLabel.text = "Predicting..."
            let vec = tfidf(sms: message!)
            do {
                //Get prediction on the text
                let prediction = try SpamClassifier().prediction(message: vec).spam_or_not
                print (prediction)
                if (prediction == "spam"){
                    outputLabel.text = "It's a SPAM!!"
                }
                else if(prediction == "ham"){
                    outputLabel.text = "Don't Worry, it's a ham!"
                }
            }
            catch{
                outputLabel.text = "No Prediction"
            }
            
        }
        
    }
    
    
    func tfidf(sms: String) -> MLMultiArray {
        //get path for files
        let wordsFile = Bundle.main.path(forResource: "wordlist", ofType: "txt")
        let smsFile = Bundle.main.path(forResource: "SMSSpamCollection", ofType: "txt")
        do {
            //read words file
            let wordsFileText = try String(contentsOfFile: wordsFile!, encoding: String.Encoding.utf8)
            var wordsData = wordsFileText.components(separatedBy: .newlines)
            wordsData.removeLast() // Trailing newline.
            //read spam collection file
            let smsFileText = try String(contentsOfFile: smsFile!, encoding: String.Encoding.utf8)
            var smsData = smsFileText.components(separatedBy: .newlines)
            smsData.removeLast() // Trailing newline.
            let wordsInMessage = sms.split(separator: " ")
            //create a multi-dimensional array
            let vectorized = try MLMultiArray(shape: [NSNumber(integerLiteral: wordsData.count)], dataType: MLMultiArrayDataType.double)
            for i in 0..<wordsData.count{
                let word = wordsData[i]
                if sms.contains(word){
                    var wordCount = 0
                    for substr in wordsInMessage{
                        if substr.elementsEqual(word){
                            wordCount += 1
                        }
                    }
                    let tf = Double(wordCount) / Double(wordsInMessage.count)
                    var docCount = 0
                    for sms in smsData{
                        if sms.contains(word) {
                            docCount += 1
                        }
                    }
                    let idf = log(Double(smsData.count) / Double(docCount))
                    vectorized[i] = NSNumber(value: tf * idf)
                } else {
                    vectorized[i] = 0.0
                }
            }
            return vectorized
        } catch {
            return MLMultiArray()
        }
    }
    

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }


}

