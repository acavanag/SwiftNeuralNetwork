//
//  main.swift
//  SwiftNeuralNetwork
//
//  Created by Andrew Cavanagh on 1/14/15.
//  Copyright (c) 2015 Tortuca Labs. All rights reserved.
//

import Foundation

var net = NeuralNet(layerScheme: [2, 4, 1])
XORTest.trainNetwork(&net, iterations: 10000)

let kThreshold = 0.1

let stdin = NSFileHandle.fileHandleWithStandardInput()
while (true) {
    print("Enter first XOR input: ")
    let first = NSString(data: stdin.availableData, encoding: NSUTF8StringEncoding)
    print("Enter second XOR input: ")
    let second = NSString(data: stdin.availableData, encoding: NSUTF8StringEncoding)
    net.propagateForward([first!.doubleValue,second!.doubleValue])
    let result = net.results()
    var response = -1
    if result[0] >= (1 - kThreshold) {
        response = 1
    } else if result[0] <= kThreshold {
        response = 0
    }
    print("Result: \(response)")
}

