/*
  CS 6023 Assignment 3. 
  Do not make any changes to the boiler plate code or the other files in the folder.
  Use cudaFree to deallocate any memory not in usage.
  Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>

__global__ void intialNodeTranlations( int* x_change, int* y_change,int T,int *trans_d){
  int id =  threadIdx.x + blockIdx.x * blockDim.x ;
  bool check= id<T;
  if(check){
    int val=3*id;
    if(trans_d[val + 1] == 0) atomicSub(&x_change[trans_d[val]], trans_d[val + 2]);
    if(trans_d[val + 1] == 2) atomicSub(&y_change[trans_d[val]], trans_d[val + 2]);
    if(trans_d[val + 1] == 1) atomicAdd(&x_change[trans_d[val]], trans_d[val + 2]);
    if(trans_d[val + 1] == 3) atomicAdd(&y_change[trans_d[val]], trans_d[val + 2]);
  }
}

void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
  /* Function for parsing input file*/

  FILE *inputFile = NULL;
  // Read the file for input. 
  if ((inputFile = fopen (fileName, "r")) == NULL) {
    printf ("Failed at opening the file %s\n", fileName) ;
    return ;
  }

  // Input the header information.
  int numMeshes ;
  fscanf (inputFile, "%d", &numMeshes) ;
  fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;
  

  // Input all meshes and store them inside a vector.
  int meshX, meshY ;
  int globalPositionX, globalPositionY; // top left corner of the matrix.
  int opacity ;
  int* currMesh ;
  for (int i=0; i<numMeshes; i++) {
    fscanf (inputFile, "%d %d", &meshX, &meshY) ;
    fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
    fscanf (inputFile, "%d", &opacity) ;
    currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
    for (int j=0; j<meshX; j++) {
      for (int k=0; k<meshY; k++) {
        fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
      }
    }
    //Create a Scene out of the mesh.
    SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ; 
    scenes.push_back (scene) ;
  }

  // Input all relations and store them in edges.
  int relations;
  fscanf (inputFile, "%d", &relations) ;
  int u, v ; 
  for (int i=0; i<relations; i++) {
    fscanf (inputFile, "%d %d", &u, &v) ;
    edges.push_back ({u,v}) ;
  }

  // Input all translations.
  int numTranslations ;
  fscanf (inputFile, "%d", &numTranslations) ;
  std::vector<int> command (3, 0) ;
  for (int i=0; i<numTranslations; i++) {
    fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
    translations.push_back (command) ;
  }
}

__global__ void resultantTranlations( int* globalCoordinatesYd, int* y_change, int V,int* globalCoordinatesXd, int* x_change){
  int id =  threadIdx.x + blockIdx.x * blockDim.x;
  bool check=id < V;
  if(check){
    globalCoordinatesXd[id] += x_change[id];
    globalCoordinatesYd[id] += y_change[id];
  }
}

__global__ void adjacencyTranlations(int sIdx, int* x_change, int* y_change, int idNode,int* hcsrD, int numEdges){
  int id =  threadIdx.x + blockIdx.x * blockDim.x ;
  bool check=id < numEdges;
  if(check){
    x_change[hcsrD[sIdx + id]] = x_change[hcsrD[sIdx + id]] + x_change[idNode];
    y_change[hcsrD[sIdx + id]] = y_change[hcsrD[sIdx + id]] + y_change[idNode];
  }
}

void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
  /* Function for writing the final png into a file.*/
  FILE *outputFile = NULL; 
  if ((outputFile = fopen (outputFileName, "w")) == NULL) {
    printf ("Failed while opening output file\n") ;
  }
  
  for (int i=0; i<frameSizeX; i++) {
    for (int j=0; j<frameSizeY; j++) {
      fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
    }
    fprintf (outputFile, "\n") ;
  }
}


__global__ void placeMesh(int* mesh,int* FinalPng_d,int* opacity_d,int frameX,int frameY,int r_node,int c_node,int id_node,int opacity,int v,int* globalCoordinatesXd,int* globalCoordinatesYd){
  int id =  threadIdx.x + blockIdx.x * blockDim.x;
  int c = id % c_node;
  int r = id / c_node;
    
  if((r > -1 && r < r_node) && (c > -1 && c < c_node)){
    int C = c + globalCoordinatesYd[id_node];
    int R = r + globalCoordinatesXd[id_node];
    
    if ((R > -1 && R < frameX) && (C > -1 && C < frameY)){
        bool check=opacity >= opacity_d[R * frameY + C];
      if(check){
        int val=R * frameY + C;
        opacity_d[val] = opacity;
        FinalPng_d[val] = mesh[r * c_node + c];
        }
    }
  }
}



int main (int argc, char **argv) {
  
  // Read the scenes into memory from File.
  const char *inputFileName = argv[1] ;
  int* hFinalPng ; 

  int frameSizeX, frameSizeY ;
  std::vector<SceneNode*> scenes ;
  std::vector<std::vector<int> > edges ;
  std::vector<std::vector<int> > translations ;
  readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
  hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;
  
  // Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

  // Basic information.
  int V = scenes.size () ;
  int E = edges.size () ;
  int numTranslations = translations.size () ;

  // Convert the scene graph into a csr.
  scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
  int *hOffset = scene->get_h_offset () ;  
  int *hCsr = scene->get_h_csr () ;
  int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
  int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
  int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
  int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
  int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
  int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

  auto start = std::chrono::high_resolution_clock::now () ;
  // Code begins here.
  // Do not change anything above this comment.

  // declaring queue for doing bfs
  
  std::queue<int> adjacent_queue;
  int *FinalPng_d, *opacity_d;
  int *globalCoordinatesXd, *globalCoordinatesXy;

  int size1 =frameSizeX * frameSizeY * sizeof(int);
  cudaMemset(&opacity_d, size1, INT_MIN);
  cudaMalloc(&opacity_d, size1);
  cudaMalloc(&FinalPng_d, size1);
  int size2=numTranslations * 3 * sizeof(int);
  int *translations_d, *trans_h = (int*)malloc(size2);
  cudaMalloc(&translations_d, size2 );
  int size3=V * sizeof(int);
  cudaMalloc(&globalCoordinatesXd, size3);
  cudaMalloc(&globalCoordinatesXy, size3);
  cudaMemcpy(globalCoordinatesXd, hGlobalCoordinatesX, size3, cudaMemcpyHostToDevice);
  cudaMemcpy(globalCoordinatesXy, hGlobalCoordinatesY, size3, cudaMemcpyHostToDevice);
  
 int i =0;
  while(i<numTranslations){
    int val=3*i;
    trans_h[val] = translations[i][0];
    trans_h[val + 1] = translations[i][1];
    trans_h[val + 2] = translations[i][2];
    i=i+1;
  }

  cudaMemcpy(translations_d, trans_h, size2, cudaMemcpyHostToDevice);
  int *hcsr_d;
  int *x_change, *y_change;
  int size4=E * sizeof(int);
  cudaMalloc(&hcsr_d, size4);
  cudaMalloc(&y_change,size3);
  cudaMalloc(&x_change, size3);
  cudaMemcpy(hcsr_d, hCsr, size4, cudaMemcpyHostToDevice);
  cudaMemset(&y_change, size3, 0);
  cudaMemset(&x_change, size3, 0);



  
  int numBlocks = (numTranslations + 512 - 1) / 512;
  intialNodeTranlations<<<numBlocks, 512>>>( x_change, y_change,numTranslations, translations_d);
  cudaFree(translations_d);

  adjacent_queue.push(0);
  while(!adjacent_queue.empty()){
    int n_id = adjacent_queue.front();
    adjacent_queue.pop();
    bool check=hOffset[n_id] != hOffset[n_id + 1];
    if(check){
      int adj=hOffset[n_id]; 
      while(adj < hOffset[n_id + 1]){
         adjacent_queue.push(hCsr[adj]);
        adj++;
      }
      numBlocks = ((hOffset[n_id + 1] - hOffset[n_id]) + 512 - 1) / 12;
      adjacencyTranlations<<<numBlocks, 512>>>( hOffset[n_id], x_change, y_change, n_id,hcsr_d, hOffset[n_id + 1] - hOffset[n_id]);
      cudaDeviceSynchronize();
    }
  }

  

  numBlocks = (V +512 - 1) / 512;

  resultantTranlations<<<(V + 512 - 1) / 512, 512>>>( globalCoordinatesXy, y_change, V,globalCoordinatesXd, x_change);
  cudaDeviceSynchronize();
  cudaFree(x_change);
  cudaFree(y_change);
  cudaFree(hcsr_d);

  int m=0;
  while(m<V){
    int *mesh, *current_mesh = hMesh[m];
    cudaMalloc(&mesh, hFrameSizeX[m] * hFrameSizeY[m] * sizeof(int));
    cudaMemcpy(mesh, current_mesh, hFrameSizeX[m] * hFrameSizeY[m] * sizeof(int), cudaMemcpyHostToDevice);
    numBlocks = (hFrameSizeX[m] * hFrameSizeY[m] + 512 - 1) / 512;
    placeMesh<<<numBlocks, 512>>>( mesh, FinalPng_d, opacity_d, frameSizeX, frameSizeY, hFrameSizeX[m], hFrameSizeY[m], m, hOpacity[m], V,globalCoordinatesXd, globalCoordinatesXy);
    cudaFree(mesh);
    m++;
  } 



  cudaMemcpy(hFinalPng, FinalPng_d, size1, cudaMemcpyDeviceToHost);
  cudaFree(globalCoordinatesXy);
  cudaFree(globalCoordinatesXd);
  cudaFree(opacity_d);
  cudaFree(FinalPng_d);
  // Do not change anything below this comment.
  // Code ends here.

  auto end  = std::chrono::high_resolution_clock::now () ;

  std::chrono::duration<double, std::micro> timeTaken = end-start;

  printf ("execution time : %f\n", timeTaken) ;
  // Write output matrix to file.
  const char *outputFileName = argv[2] ;
  writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ; 

}

