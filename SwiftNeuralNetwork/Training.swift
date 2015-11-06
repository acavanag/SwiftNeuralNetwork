//
//  Training.swift
//  ACNeuralNetwork
//
//  Created by Andrew Cavanagh on 12/19/14.
//  Copyright (c) 2014 Tortuca Labs. All rights reserved.
//

import Foundation

typealias TrainingSet = (input: (first: Double, second: Double), output: Double)

private let kRange = 0.1

final class XORTest {
    
    private class func trainingData(iterations: Int = 10000) -> [TrainingSet] {
        var data = [TrainingSet]()
        for var i = 0; i < iterations; i++ {
            let firstRandomBit = rand() & 1
            let secondRandomBit = rand() & 1
            let output = firstRandomBit ^ secondRandomBit
            
            let trainingSet: TrainingSet = ((Double(firstRandomBit), Double(secondRandomBit)), Double(output))
            data.append(trainingSet)
        }
        return data
    }
    
    private class func checkResults(results: [Double], trainingSet: TrainingSet, iteration: Int, error: Double = 0) {
        let largest = (results[0] >= trainingSet.output) ? results[0] : trainingSet.output
        var smallest: Double!
        if largest == results[0] {
            smallest = trainingSet.output
        } else {
            smallest = results[0]
        }
        
        let correct = (abs(largest - smallest) <= kRange) ? "Correct" : "Incorrect"
        
        print("Iteration \(iteration) [inputs: (\(trainingSet.input.first) , \(trainingSet.input.second))] -> [output: \(results[0].format())] [Error: \(error.format())] = \(correct)")
    }
    
    class func trainNetwork(inout net: NeuralNet, iterations: Int) {
        let trainData = self.trainingData(iterations)
        var index: Int = 0
        for trainSet in trainData {
            net.propagateForward([trainSet.input.first, trainSet.input.second])
            checkResults(net.results(), trainingSet: trainSet, iteration: index, error: net.error)
            net.propagateBackward([trainSet.output])
            index++
        }
    }
    
}

extension Double {
    func format() -> String {
        return NSString(format: "%.13f", self) as String
    }
}
