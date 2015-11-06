//
//  NeuralNet.swift
//  ACNN2
//
//  Created by Andrew Cavanagh on 1/9/15.
//  Copyright (c) 2015 Andrew Cavanagh. All rights reserved.
//

import Foundation

final class NeuralNet {

    private final var layers = [Layer]()
    final var error: Double = 0
    
    init(layerScheme: [Int]) {
        for proposedNeuronCount in layerScheme {
            layers.append(Layer(neuronCount: proposedNeuronCount))
        }
        for var i = 0; i < layers.count - 1; i++ {
            layers[i].projectToLayer(layers[i + 1])
        }
    }
    
    // MARK: - Forward Propogation
    
    func propagateForward(inputValues: [Double]) {
        for (index, _) in inputValues.enumerate() {
            layers[0].neurons[index].outputValue = inputValues[index]
        }
        for index in 1 ..< layers.count {
            layers[index].propagateForward(layers[index - 1])
        }
    }
    
    // MARK: - Backwards Propogation
    
    func propagateBackward(targetValues: [Double]) {
        error = 0
        let outputLayer = layers[layers.count - 1]
        let outputLayerNeuronCount = outputLayer.neurons.count - 1
        for var i = 0; i < outputLayerNeuronCount; i++ {
            let delta = targetValues[i] - outputLayer.neurons[i].outputValue
            error = error + (delta * delta)
        }
        error = sqrt(error / Double(outputLayerNeuronCount))
        
        for var i = 0; i < outputLayerNeuronCount; i++ {
            outputLayer.neurons[i].calculateOutputGradientDescent(targetValues[i])
        }
        
        for var layerIndex = layers.count - 2; layerIndex > 0; layerIndex-- {
            let currentLayer = layers[layerIndex]
            let nextLayer = layers[layerIndex + 1]
            
            for var i = 0; i < currentLayer.neurons.count; i++ {
                currentLayer.neurons[i].calculateHiddentGradientDescent(nextLayer)
            }
        }
        
        for var layerIndex = layers.count - 1; layerIndex > 0; layerIndex-- {
            let currentLayer = layers[layerIndex]
            let previousLayer = layers[layerIndex - 1]
            
            for var i = 0; i < currentLayer.neurons.count - 1; i++ {
                currentLayer.neurons[i].updateInputWeights(previousLayer)
            }
        }
    }
    
    // MARK: - Network Results
    
    func results() -> [Double] {
        var results = [Double]()
        let outputNeurons = layers[layers.count - 1].neurons
        for var i = 0; i < outputNeurons.count - 1; i++ {
            results.append(outputNeurons[i].outputValue)
        }
        return results
    }
    
    // MARK: - Projection
    
    private struct Projection {
        var weight: Double = 0
        var deltaWeight: Double = 0
    }
    
    // MARK: - Neuron
    
    private final class Neuron {
        private final var gradient: Double = 0
        private final var outputValue: Double = 1
        private final var index: Int = 0
        private final var projections = [Projection]()
        
        init(positionInLayer: Int) {
            index = positionInLayer
        }
        
        func propagateForward(previousLayer: Layer) {
            var sum: Double = 0.0
            for neuron in previousLayer.neurons {
                sum = sum + (neuron.outputValue * neuron.projections[index].weight)
            }
            outputValue = Utils.transfer(sum)
        }
        
        func calculateOutputGradientDescent(targetValue: Double) {
            let delta = targetValue - outputValue
            gradient = delta * Utils.transferDerivative(outputValue)
        }
        
        func calculateHiddentGradientDescent(nextLayer: Layer) {
            var sum: Double = 0.0
            for var i = 0; i < nextLayer.neurons.count - 1; i++ {
                sum = sum + (projections[i].weight * nextLayer.neurons[i].gradient)
            }
            gradient = sum * Utils.transferDerivative(outputValue)
        }
        
        func updateInputWeights(previousLayer: Layer) {
            for neuron in previousLayer.neurons {
                let oldDeltaWeight = neuron.projections[index].deltaWeight
                let newDeltaWeight = (Utils.kEta * neuron.outputValue * gradient) + (Utils.kAlpha * oldDeltaWeight)
                neuron.projections[index].deltaWeight = newDeltaWeight
                neuron.projections[index].weight = neuron.projections[index].weight + newDeltaWeight
            }
        }
    }
    
    // MARK: - Layer
    
    private final class Layer {
        private final var neurons = [Neuron]()
        init(neuronCount: Int) {
            for var i = 0; i < neuronCount; i++ {
                neurons.append(Neuron(positionInLayer: i))
            }
            neurons.append(Neuron(positionInLayer: neuronCount))
        }
        
        func projectToLayer(forwardLayer: Layer) {
            for neuron in neurons {
                for _ in forwardLayer.neurons {
                    neuron.projections.append(Projection(weight: Utils.randomWeight(), deltaWeight: 0))
                }
            }
        }
        
        func propagateForward(previousLayer: Layer) {
            for var i = 0; i < neurons.count - 1; i++ {
                neurons[i].propagateForward(previousLayer)
            }
        }
    }
    
    // MARK: - Utilities
    
    private struct Utils {
        
        static let kEta: Double = 0.15
        static let kAlpha: Double = 0.5
        
        static func randomWeight() -> Double {
            return Double(rand()) / Double(RAND_MAX)
        }
        
        static func transfer(x: Double) -> Double {
            return tanh(x)
        }
        
        static func transferDerivative(x: Double) -> Double {
            return x * (1.0 - x)
        }
    }

}