//
//  ViewController.swift
//  ARKit-CoreML
//
//  Created by cl-dev on 2018-04-03.
//  Copyright © 2018 cl-dev. All rights reserved.
//

import UIKit
import SceneKit
import ARKit
import Vision

class ViewController: UIViewController, ARSCNViewDelegate {

  @IBOutlet var sceneView: ARSCNView!

  lazy var mlModel: VNCoreMLModel! = {
    do {
      return try VNCoreMLModel(for: MobileNet().model)
    } catch {
      fatalError("Could not load MLModel")
    }
  }()

  let mlDispatchQueue = DispatchQueue(label: "com.connectedlabs.mlqueue")

  lazy var predictionLabel: UILabel = {
    let label = UILabel()
    return label
  }()

  lazy var saveButton: UIButton = {
    let button = UIButton(type: .system)
    button.setTitle("Save Prediction", for: .normal)
    button.addTarget(self, action: #selector(saveCurrentPrediction), for: .touchUpInside)
    return button
  }()

  var currentPrediction = "" {
    didSet {
      self.predictionLabel.text = currentPrediction
    }
  }

  override func viewDidLoad() {
    super.viewDidLoad()

    // Set the view's delegate
    sceneView.delegate = self

    // Show statistics such as fps and timing information
    sceneView.showsStatistics = true

    // Create a new scene
    let scene = SCNScene()

    // Set the scene to the view
    sceneView.scene = scene

    predictionLabel.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(predictionLabel)
    predictionLabel.leftAnchor.constraint(equalTo: view.leftAnchor).isActive = true
    predictionLabel.rightAnchor.constraint(equalTo: view.rightAnchor).isActive = true
    predictionLabel.topAnchor.constraint(equalTo: view.topAnchor, constant: 20).isActive = true

    saveButton.translatesAutoresizingMaskIntoConstraints = false
    view.addSubview(saveButton)
    saveButton.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -20).isActive = true
    saveButton.rightAnchor.constraint(equalTo: view.rightAnchor).isActive = true
  }

  override func viewWillAppear(_ animated: Bool) {
    super.viewWillAppear(animated)

    // Create a session configuration
    let configuration = ARWorldTrackingConfiguration()

    // Run the view's session
    sceneView.session.run(configuration)

    startMLQueue()
  }

  override func viewWillDisappear(_ animated: Bool) {
    super.viewWillDisappear(animated)

    // Pause the view's session
    sceneView.session.pause()
  }

  //////////////////////////////////////////////////////////
  ///////////////////// MARK: - CoreML /////////////////////
  //////////////////////////////////////////////////////////

  func startMLQueue() {
    var ciImage: CIImage? = nil
    defer {
      mlDispatchQueue.async { [weak self] in
        // create a prediction request then queue up another dispatch
        if let ciImage = ciImage {
          self?.createPredictionRequest(ciImage)
        }

        self?.startMLQueue()
      }
    }
    guard let pixelBuffer = sceneView?.session.currentFrame?.capturedImage else {
      return
    }
    ciImage = CIImage(cvPixelBuffer: pixelBuffer)
  }

  func createPredictionRequest(_ ciImage: CIImage) {
    let imagePredictionRequest = VNCoreMLRequest(model: mlModel, completionHandler: predictionRequestCompletionHandler)
    let imagePredictionHandler = VNImageRequestHandler(ciImage: ciImage, options: [:])
    try? imagePredictionHandler.perform([imagePredictionRequest])
  }

  func predictionRequestCompletionHandler(request: VNRequest, error: Error?) {
    guard error == nil, let results = request.results as? [VNClassificationObservation] else {
      return
    }

    let topResult = results[0]
    DispatchQueue.main.async {
      self.currentPrediction = topResult.identifier
    }
  }

  //////////////////////////////////////////////////////////
  ///////////////////// MARK: - ARKit  /////////////////////
  //////////////////////////////////////////////////////////

  let bubbleDepth: Float = 0.01

  @objc func saveCurrentPrediction() {
    let sceneCenter = CGPoint(x: self.sceneView.bounds.midX, y: self.sceneView.bounds.midY)
    let arHitTestResults = sceneView.hitTest(sceneCenter, types: .featurePoint)
    guard let closestResult = arHitTestResults.first else {
      return
    }

    let transform = closestResult.worldTransform
    let worldCoord = SCNVector3Make(transform.columns.3.x, transform.columns.3.y, transform.columns.3.z)

    let node = createNewBubbleParentNode(currentPrediction)
    sceneView.scene.rootNode.addChildNode(node)
    node.position = worldCoord
  }

  func createNewBubbleParentNode(_ text : String) -> SCNNode {
    // Warning: Creating 3D Text is susceptible to crashing. To reduce chances of crashing; reduce number of polygons, letters, smoothness, etc.

    // TEXT BILLBOARD CONSTRAINT
    let billboardConstraint = SCNBillboardConstraint()
    billboardConstraint.freeAxes = SCNBillboardAxis.Y

    // BUBBLE-TEXT
    let bubble = SCNText(string: text, extrusionDepth: CGFloat(bubbleDepth))
    let font = UIFont.systemFont(ofSize: 0.15)
//    var font = UIFont(name: "Futura", size: 0.15)
    bubble.font = font
    bubble.alignmentMode = kCAAlignmentCenter
    bubble.firstMaterial?.diffuse.contents = UIColor.blue
    bubble.firstMaterial?.specular.contents = UIColor.white
    bubble.firstMaterial?.isDoubleSided = true
    // bubble.flatness // setting this too low can cause crashes.
    bubble.chamferRadius = CGFloat(bubbleDepth)

    // BUBBLE NODE
    let (minBound, maxBound) = bubble.boundingBox
    let bubbleNode = SCNNode(geometry: bubble)
    // Centre Node - to Centre-Bottom point
    bubbleNode.pivot = SCNMatrix4MakeTranslation( (maxBound.x - minBound.x)/2, minBound.y, bubbleDepth/2)
    // Reduce default text size
    bubbleNode.scale = SCNVector3Make(0.2, 0.2, 0.2)

    // CENTRE POINT NODE
    let sphere = SCNSphere(radius: 0.005)
    sphere.firstMaterial?.diffuse.contents = UIColor.cyan
    let sphereNode = SCNNode(geometry: sphere)

    // BUBBLE PARENT NODE
    let bubbleNodeParent = SCNNode()
    bubbleNodeParent.addChildNode(bubbleNode)
    bubbleNodeParent.addChildNode(sphereNode)
    bubbleNodeParent.constraints = [billboardConstraint]

    return bubbleNodeParent
  }


}
