#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

int main() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    MPSMatrixDecompositionCholesky *chol = [[MPSMatrixDecompositionCholesky alloc] initWithDevice:device lower:YES order:10];
    return 0;
}
