//
//  ImagerRenderer.swift
//  RTSwift
//
//  Created by 徐浩博 on 2021/2/6.
//

import Foundation

class ImageRenderer {
    
    var imageWidth: Int
    var imageHeight: Int
    var aspectRatio: Float;
    var bytesPerRow: Int
    var byteCount: Int
    var raw: UnsafeMutableRawPointer
    var data: UnsafeMutableRawBufferPointer
    
    init(imageWidth: Int, imageHeight: Int) {
        self.imageWidth = imageWidth //1280
        self.imageHeight = imageHeight //720
        self.aspectRatio = Float(imageWidth) / Float(imageHeight) // 16:9
        
        bytesPerRow = imageWidth * 4
        byteCount = bytesPerRow * imageHeight
        
        raw = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: 1)
        data = UnsafeMutableRawBufferPointer(start: raw, count: byteCount)
    }
    
    deinit {
        raw.deallocate()
    }
    
    func setColor(x: Int, y: Int, color8Bit: Color8Bit) {
        // origin at bottom left
        let index = ((imageHeight - y - 1) * imageWidth + x) * 4
        
        // set pixel color to data array
        data[index]     = color8Bit.r
        data[index + 1] = color8Bit.g
        data[index + 2] = color8Bit.b
    }
    
    func savePNGImage(imageName: String, rootURL: String) {
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
            print("CGColorSpace create failed")
            return
        }
        let typedData = CFDataCreate(nil, raw.assumingMemoryBound(to: UInt8.self),
                                     byteCount)
        guard let dataProvider = CGDataProvider(data: typedData!) else {
            print("CGDataProvider create failed")
            return
        }
        
        let img = CGImage(width: imageWidth, height: imageHeight, bitsPerComponent: 8,
                          bitsPerPixel: 32, bytesPerRow: bytesPerRow,
                          space: colorSpace,
                          bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                          provider: dataProvider, decode: nil,
                          shouldInterpolate: false,
                          intent: CGColorRenderingIntent.defaultIntent)
        guard let imgUnwrapped = img else {
            print("CGImage create failed")
            return
        }
        let urlStr = rootURL + "/" + imageName
        let url = URL(fileURLWithPath: urlStr)
        let imgDest = CGImageDestinationCreateWithURL(url as CFURL,
                                                      "public.png" as CFString,
                                                      1, nil)
        guard let imgDestUnwrapped = imgDest else {
            print("CGImageDestination create failed")
            return
        }
        
        CGImageDestinationAddImage(imgDestUnwrapped, imgUnwrapped, nil)
        CGImageDestinationFinalize(imgDestUnwrapped)
    }
}
