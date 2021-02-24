#include <stdarg.h>
#include <cstdio>
#include <sys/time.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <sstream>
#define MPI_USED
#define OMP_USED
#ifdef MPI_USED
#include <mpi.h>
#endif
#ifdef OMP_USED
#include <omp.h>
#endif
using namespace std;

struct Topo {
 int N,Nz;
 vector<int> IA, JA;
};

class Sorter {
public:
   Sorter(const vector<int>& l2g, const vector<int>& part, int myid) : L2G(l2g), Part(part), myID(myid) {}
   bool operator()(int,int);
private:
   const vector<int> &L2G, &Part;
   int myID;
};

bool Sorter::operator()(int elem1, int elem2) {
   if (Part[elem1] == Part[elem2])
      if (L2G[elem1] < L2G[elem2])
          return true;
      else
          return false;
   else
      if (Part[elem1] == myID)
	  return true;
      else
	  if (Part[elem2] == myID)
              return false;
          else
	      return (Part[elem1] < Part[elem2]);
}

class Mesh {
public:
 Mesh() : NE (NULL), EN (NULL), EE (NULL), 
#ifdef MPI_USED
		in_scheme(NULL), out_scheme(NULL), 
#endif
			vec(NULL), myID(0) {}
 ~Mesh();
 void generate(int nx,int ny, int K3, int K4, int,int);
 void print(int P);
 void printRes(int, int, int, int);
 void printControlSums();
 void SpMV();
#ifdef MPI_USED
 void refreshHalo();
#endif
private:
 enum Mode {Left,Right,Up,Down,Centre};
 int addRect(int curr,int width,int height,int Nx,int Ny,int K3,int K4,
		 int vertical_offset,int horizontal_offset, Mode mode, int px);
 vector<int> getElementVertices(int nx, int ny, int curr, bool ntd = true, Mode mode = Centre) const;
 inline int gElementNumber(int currGlobal,int K3,int K4) const;
 inline int gGlobalRectNum(int Nx,int vertical_offset,int curr,int width,int horizontal_offset) const;
 inline bool needToDivide(int,int,int);
 void ENtoNE();
 void toEE();
 void fill();
#ifdef MPI_USED
 void createChangesScheme();
 Topo *in_scheme, *out_scheme;
 vector<int> neighbours;
 double* out_message;
 MPI_Request* req;
#endif

 int myID;
 Topo *NE, *EN, *EE;
 vector<int> L2G, Part;
 vector<double> mat,res;
 double *vec;
};

void Mesh::SpMV() {
    #pragma omp parallel for 
        for (int i = 0; i < EE->N; i++) {
	       double sum = 0;
               for (int j = EE->IA[i]; j< EE->IA[i+1]; j++)
                   sum+=mat[j]*vec[EE->JA[j]];
	       res[i] = sum;
        }
}

int Mesh::addRect(int curr,int width,int height,int Nx,int Ny,int K3,int K4,
				 int vertical_offset,int horizontal_offset, Mode mode, int px) {
    int currGlobal, elNum;
    vector<int> vertices, localVertices;
    currGlobal = gGlobalRectNum(Nx,vertical_offset,curr,width,horizontal_offset);
    switch (mode)
    {
       case Up:
          currGlobal -= Nx;
	  break;
       case Down:
          currGlobal += Nx;
	  break;
       case Left:
          currGlobal -= 1;
	  break;
       case Right:
          currGlobal += 1;
	  break;
    }
    bool need_to_divide = needToDivide(currGlobal, K3, K4);
    vertices = getElementVertices(Nx, Ny, currGlobal);
    localVertices = getElementVertices(width, height, curr, !need_to_divide, mode);
    elNum = gElementNumber(currGlobal,K3,K4);
    if (need_to_divide) {
    //divide into triangles
      if (mode == Centre || mode == Right || mode == Down) {
         EN->N+=1;
         EN->Nz+=3;
         EN->IA.push_back(EN->IA.back()+3);
         L2G.push_back(elNum);
         EN->JA.push_back(localVertices[0]);
         EN->JA.push_back(localVertices[1]);
         EN->JA.push_back(localVertices[3]);
         switch (mode)
         {
            case Centre:
               Part.push_back(myID);
     	       break;
            case Down:
               Part.push_back(myID + px);
     	       break;
            case Right:
               Part.push_back(myID + 1);
     	       break;
         }
      }
      if (mode == Centre || mode == Left || mode == Up) {
         EN->N+=1;
         EN->Nz+=3;
         EN->IA.push_back(EN->IA.back()+3);
         L2G.push_back(elNum+1);
         EN->JA.push_back(localVertices[1]);
         EN->JA.push_back(localVertices[2]);
         EN->JA.push_back(localVertices[3]);
         switch (mode)
         {
            case Centre:
               Part.push_back(myID);
     	       break;
            case Up:
               Part.push_back(myID - px);
     	       break;
            case Left:
               Part.push_back(myID - 1);
     	       break;
         }
      }
    }
    else {
      EN->N+=1;
      EN->Nz+=4;
      EN->IA.push_back(EN->IA.back()+4);
      L2G.push_back(elNum);
      EN->JA.push_back(localVertices[0]);
      EN->JA.push_back(localVertices[1]);
      EN->JA.push_back(localVertices[2]);
      EN->JA.push_back(localVertices[3]);
      switch (mode)
      {
         case Centre:
            Part.push_back(myID);
            break;
         case Down:
            Part.push_back(myID + px);
            break;
         case Right:
            Part.push_back(myID + 1);
            break;
         case Up:
            Part.push_back(myID - px);
            break;
         case Left:
            Part.push_back(myID - 1);
            break;
      }
    }
    switch (mode)
    {
       case Centre:
          return 0;
       case Down:
          return 1 + ((curr == width*height - 1) && !need_to_divide? 1 : 0);
       case Right:
          return 1 + ((curr == width*height - 1) && !need_to_divide? 1 : 0);
       case Up:
          return 1 + ((curr == 0) && !need_to_divide? 1 : 0);
       case Left:
          return 1 + ((curr == 0) && !need_to_divide? 1 : 0);
    }
}

vector<int> Mesh::getElementVertices(int nx, int ny, int curr, bool is_square, Mode mode) const
{
  vector<int> result;
  switch (mode) {
     case Right:
         result.push_back(curr/nx*(nx+1)+nx);
         result.push_back(NE->N);
         result.push_back(result.back()+1);
         result.push_back(result.front()+nx+1);
	 break;
     case Up:
         result.push_back(NE->N - (curr == 0 && is_square ? 0 : 1));
         result.push_back(result.back()+1);
         result.push_back(curr+1);
         result.push_back(curr);
	 break;
     case Down:
         result.push_back(ny*(nx+1)+curr%nx);
         result.push_back(result.back()+1);
         result.push_back(NE->N+1);
         result.push_back(result.back()-1);
	 break;
     case Left:
         result.push_back(NE->N - (curr == 0 && is_square ? 0 : 1));
         result.push_back(curr/nx*(nx+1));
         result.push_back(result.back()+nx+1);
         result.push_back(result.front()+1);
	 break;
     case Centre:
         result.push_back(curr/nx*(nx+1)+curr%nx);
         result.push_back(result.back()+1);
         result.push_back(result.front()+nx+2);
         result.push_back(result.back()-1);
	 break;
  }
  return result;
}

inline int Mesh::gElementNumber(int currGlobal,int K3,int K4) const
{
    return currGlobal/(K3+K4)*(2*K3 + K4) + (currGlobal%(K3+K4)<K3?currGlobal%(K3+K4)*2:K3*2+currGlobal%(K3+K4)-K3);
}

inline int Mesh::gGlobalRectNum(int Nx,int vertical_offset,int curr,int width,int horizontal_offset) const
{
    return Nx * (vertical_offset + curr/width) + horizontal_offset + curr % width;
}

inline bool Mesh::needToDivide(int currGlobal, int K3, int K4)
{
    return currGlobal%(K3+K4) < K3;
}

int pprintf(const char *fmt,...){
  int myID = 0; // process ID
  int P = 1; // number of processes
  int r = 0;
#ifdef MPI_USED
    int mpi_res;
    mpi_res = MPI_Comm_rank(MPI_COMM_WORLD,&myID);
    mpi_res = MPI_Comm_size(MPI_COMM_WORLD,&P);

    char io_buf[256];
    MPI_Status status;
    if (myID==0) {
        va_list ap;
        va_start(ap,fmt);
        r=vfprintf(stdout,fmt,ap);
        va_end(ap);
        for(int p=1; p<P; p++){
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Recv(io_buf,256,MPI_CHAR,p,0,MPI_COMM_WORLD,&status);
            fprintf(stdout,"%s",io_buf);
        }
    }
    else
        for(int p=1; p<P; p++){
            MPI_Barrier(MPI_COMM_WORLD);
            if(myID != p) continue;
            va_list ap;
            va_start(ap,fmt);
            r=vsprintf(io_buf,fmt,ap);
            va_end(ap);
            MPI_Send(io_buf,strlen(io_buf)+1,MPI_CHAR,0,0,MPI_COMM_WORLD);
        }
    MPI_Barrier(MPI_COMM_WORLD);
#else
        va_list ap;
        va_start(ap,fmt);
        r=vfprintf(stdout,fmt,ap);
        va_end(ap);
#endif
    return(r);
}


Mesh::~Mesh() {
  if (EN!=NULL) {
    delete NE;
    delete EN;
    delete EE;
  }
#ifdef MPI_USED
  if (in_scheme) {
    delete in_scheme;
    delete out_scheme;
    delete[] out_message;
    delete[] req;
  }
#endif
    delete[] vec;
}


void Mesh::generate(int Nx,int Ny, int K3, int K4, int px, int py) {
  int P = px * py; // number of processes 
#ifdef MPI_USED
  int mpi_res;
  mpi_res = MPI_Comm_rank(MPI_COMM_WORLD,&myID);
#endif
  EN = new Topo;
  NE = new Topo;
  EN->N = 0;
  EN->Nz = 0;
  EN->IA.push_back(0);
  int width = Nx/px + (myID%px>Nx-Nx/px*px-1?0:1);
  int max_width = Nx<=px*width?width:width+1;
  int amount_of_bigs_x = Nx - px * (max_width - 1);
  int height = Ny/py + (myID/px>Ny-Ny/py*py-1?0:1);
  int max_height = Ny<=py*height?height:height+1;
  int amount_of_bigs_y = Ny - py * (max_height - 1);
  int horizontal_offset = width<max_width?amount_of_bigs_x*max_width+(myID%px-amount_of_bigs_x)*width:myID%px*width;
  int vertical_offset = height<max_height?amount_of_bigs_y*max_height+(myID/px-amount_of_bigs_y)*height:myID/px*height;
  for(int curr=0; curr<width*height; curr++) {
    addRect(curr,width,height,Nx,Ny,K3,K4,vertical_offset,horizontal_offset,Centre,px);
  }
  NE->N = (width+1)*(height+1);
  //add halo
  if (vertical_offset > 0) {
      for (int i = 0; i<width; i++)
         NE->N += addRect(i,width,height,Nx,Ny,K3,K4,vertical_offset,horizontal_offset,Up,px);
  }
  if (horizontal_offset > 0) {
      for (int i = 0; i<height; i++)
         NE->N += addRect(i*width,width,height,Nx,Ny,K3,K4,vertical_offset,horizontal_offset,Left,px);
  }
  if (width + horizontal_offset < Nx) {
      for (int i = 0; i<height; i++)
         NE->N += addRect((i+1)*width-1,width,height,Nx,Ny,K3,K4,vertical_offset,horizontal_offset,Right,px);
  }
  if (vertical_offset + height < Ny) {
      for (int i = 0; i<width; i++)
         NE->N += addRect(width*(height-1)+i,width,height,Nx,Ny,K3,K4,vertical_offset,horizontal_offset,Down,px);
  }
  ENtoNE();
  toEE();
  fill();
#ifdef MPI_USED
  if (px*py > 1)
    createChangesScheme();
#endif
}

void Mesh::ENtoNE() {
  NE->IA.assign(NE->N+1,0);
  for (std::vector<int>::iterator it = EN->JA.begin(); it != EN->JA.end(); ++it)
    NE->IA[*it+1]++;
  for (int i=1; i<NE->N+1 ;i++)
    NE->IA[i]+=NE->IA[i-1];
  NE->Nz = EN->Nz;
  NE->JA.assign(NE->Nz,0);
  for (int i=0, currElem=0; i<EN->Nz; i++) {
    int offset;
    offset = NE->JA[NE->IA[EN->JA[i]+1]-1];
    NE->JA[NE->IA[EN->JA[i]]+offset] = currElem;
    if(NE->IA[EN->JA[i]]+offset+1<NE->IA[EN->JA[i]+1])
      NE->JA[NE->IA[EN->JA[i]+1]-1]++;
    if (i==EN->IA[currElem+1]-1)
      currElem++;
  }
}

void Mesh::toEE() {
   EE = new Topo;
   EE->N = 0;
   EE->Nz = 0;
   EE->IA.push_back(0);
   for (int elem_num = 1, counter = 0; elem_num <= EN->N && Part[elem_num-1] == myID; elem_num++) {
      vector<int>::iterator left_iterator, right_iterator;
      (EE->N)++;
      EE->IA.push_back(EE->IA.back() + 1);
      EE->JA.push_back(elem_num - 1);
      (EE->Nz)++;
      int first_node, second_node;
      for (;counter < EN->IA[elem_num]; counter++) {
          first_node = EN->JA[counter];
	  second_node = EN->JA[(counter+1 < EN->IA[elem_num]?counter+1:EN->IA[elem_num-1])];
	  for (int counter2 = NE->IA[second_node]; counter2 < NE->IA[second_node+1]; counter2++) {
              int elem2 = NE->JA[counter2];
	      int counter3;
	      for(counter3 = EN->IA[elem2]; EN->JA[counter3]!=second_node; counter3++) {}
              if (EN->JA[(counter3+1 < EN->IA[elem2+1]?counter3+1:EN->IA[elem2])] == first_node) {
                  (EE->Nz)++;
		  (EE->IA.back())++;
		  EE->JA.push_back(elem2);
              }
          }
      }
      Sorter sorter(L2G, Part, myID);
      left_iterator = EE->JA.begin() + EE->IA[EE->N-1];
      right_iterator = EE->JA.begin() + EE->IA[EE->N];
      sort(left_iterator, right_iterator, sorter);
   }
}

void Mesh::fill() {
    vec = new double[EN->N];
    for(int row = 0; row < EE->N; row++) {
        for (vector<int>::iterator col = EE->JA.begin() + EE->IA[row]; col != EE->JA.begin() + EE->IA[row+1]; col++)
            mat.push_back(sin(L2G[row])+cos(L2G[*col]));
	vec[row] = sin(pow((double)L2G[row],2));
    }
    res.assign(EE->N, 0);
}

#ifdef MPI_USED
void Mesh::createChangesScheme() {
   for(int elem = EE->N; elem<EN->N; elem++)
       if (Part[elem]!=Part[elem-1])
           neighbours.push_back(Part[elem]);
   in_scheme = new Topo;
   in_scheme->N = neighbours.size();
   in_scheme->IA.resize(in_scheme->N+1);
   in_scheme->IA[0] = 0;
   out_scheme = new Topo;
   out_scheme->N = neighbours.size();
   out_scheme->Nz = 0;
   out_scheme->IA.resize(out_scheme->N+1);
   out_scheme->IA[0] = 0;
   vector<vector<bool> > need_to_add(neighbours.size());
   for (int i = 0; i<neighbours.size(); i++) {
       need_to_add[i].assign(EE->N,false);
   }
   for(int elem=EE->N, i=0; elem<EN->N; elem++) {
      if (Part[elem]!=Part[elem-1]) {
	  i++;
	  in_scheme->IA[i] = in_scheme->IA[i-1];
      }
      in_scheme->IA[i]++;
   }
   for(int elem=0; elem<EE->N; elem++)
      for (int j = EE->IA[elem]; j< EE->IA[elem+1];j++) {
          int elem2 = EE->JA[j], owner;
	  if ((owner = Part[elem2]) != myID) {
	      int j;
              for(j = 0; j<neighbours.size() && owner!=neighbours[j]; j++) {}
	      need_to_add[j][elem] = true;
	  }
      }
   for(int j = 0; j<neighbours.size(); j++) {
       out_scheme->IA[j+1] = out_scheme->IA[j];
       for(int elem=0; elem<EE->N; elem++)
            if (need_to_add[j][elem]) {
                  out_scheme->IA[j+1]++;
                  out_scheme->Nz++;
                  out_scheme->JA.push_back(elem);
	    }
   }
   out_message = new double[out_scheme->Nz];
   req = new MPI_Request[2*neighbours.size()];
}

void Mesh::refreshHalo() {
    double *in_message = vec + EE->N;
    int i,j;

    for (i = 0; i < out_scheme->N; i++)
        for(j = out_scheme->IA[i]; j < out_scheme->IA[i+1]; j++)
           out_message[j] = vec[out_scheme->JA[j]];

//    for (i = 0; i<neighbours.size()?(neighbours[i] < myID):false; i++)
//       MPI_Recv(in_message + in_scheme->IA[i], in_scheme->IA[i+1] - in_scheme->IA[i], 
//		       MPI_DOUBLE, neighbours[i], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//    for (; i<neighbours.size(); i++)
//       MPI_Send(out_message + out_scheme->IA[i], out_scheme->IA[i+1] - out_scheme->IA[i], 
//		       MPI_DOUBLE, neighbours[i], 0, MPI_COMM_WORLD);
//    for (i = 0; i<neighbours.size()?(neighbours[i] < myID):false; i++)
//       MPI_Send(out_message + out_scheme->IA[i], out_scheme->IA[i+1] - out_scheme->IA[i], 
//		       MPI_DOUBLE, neighbours[i], 0, MPI_COMM_WORLD);
//    for (; i<neighbours.size(); i++)
//       MPI_Recv(in_message + in_scheme->IA[i], in_scheme->IA[i+1] - in_scheme->IA[i], 
//		       MPI_DOUBLE, neighbours[i], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (i = 0; i<neighbours.size(); i++)
       MPI_Isend(out_message + out_scheme->IA[i], out_scheme->IA[i+1] - out_scheme->IA[i], 
		       MPI_DOUBLE, neighbours[i], 0, MPI_COMM_WORLD, req+i);
    for (i = 0; i<neighbours.size(); i++)
       MPI_Irecv(in_message + in_scheme->IA[i], in_scheme->IA[i+1] - in_scheme->IA[i], 
		       MPI_DOUBLE, neighbours[i], 0, MPI_COMM_WORLD, req+i+neighbours.size());
    MPI_Waitall(2*neighbours.size(), req, MPI_STATUSES_IGNORE);

}
#endif

void Mesh::printRes(int Nx, int Ny, int K3, int K4) {
     int totalElementsAmount = gElementNumber(Nx*Ny, K3, K4);
//     int current = 0;
//     stringstream s;
//     for (int i=0; i<totalElementsAmount; i++) {
//         s.clear();
//	 s.str(string());
//         if (L2G[current] == i && current < EE->N)
//	     s<<res[current++]<<"     ";
//	 pprintf(s.str().c_str());
//     }
     if (myID == 0)
	 cout<<"Total Elements Amount  "<< totalElementsAmount<<endl;
}

void Mesh::print(int P) {
  for (int i=0; i<P;i++) {
     if (myID == i) {
         cout<<"-------------------------Process "<<myID<<"------------------------------------"<<endl;
         cout<<endl<<"E->N     IA   " << EN->N <<endl;
         for (std::vector<int>::iterator it = EN->IA.begin(); it != EN->IA.end(); ++it)
           cout<<*it<<' ';
         cout<<endl<<"E->N     JA   " << EN->Nz <<endl;
         for (std::vector<int>::iterator it = EN->JA.begin(); it != EN->JA.end(); ++it)
         cout<<(*it)<<' ';
         cout<<endl<<"N->E     IA   " << NE->N <<endl;
         for (std::vector<int>::iterator it = NE->IA.begin(); it != NE->IA.end(); ++it)
           cout<<*it<<' ';
         cout<<endl<<"N->E     JA   " << NE->Nz <<endl;
         for (std::vector<int>::iterator it = NE->JA.begin(); it != NE->JA.end(); ++it)
           cout<<*it<<' ';
         cout<<endl<<"L2G"<<endl;
         for (std::vector<int>::iterator it = L2G.begin(); it != L2G.end(); ++it)
           cout<<*it<<' ';
         cout<<endl<<"Part"<<endl;
         for (std::vector<int>::iterator it = Part.begin(); it != Part.end(); ++it)
           cout<<*it<<' ';
         cout<<endl<<"E->E     IA   " << EE->N <<endl;
         for (std::vector<int>::iterator it = EE->IA.begin(); it != EE->IA.end(); ++it)
           cout<<*it<<' ';
         cout<<endl<<"E->E     JA   " << EE->Nz <<endl;
         for (std::vector<int>::iterator it = EE->JA.begin(); it != EE->JA.end(); ++it)
           cout<<*it<<' ';
         cout<<endl<<"matrix   " << EE->Nz <<endl;
         for (std::vector<double>::iterator it = mat.begin(); it != mat.end(); ++it)
           cout<<*it<<' ';
//         cout<<endl<<"vec      " << vec.size() <<endl;
//         for (std::vector<double>::iterator it = vec.begin(); it != vec.end(); ++it)
//           cout<<*it<<' ';
         cout<<endl;
      }
#ifdef MPI_USED
      MPI_Barrier(MPI_COMM_WORLD);
#endif
   }
}

void Mesh::printControlSums() {
    double sum = 0, out;
    for(int i = 0; i<res.size();i++)
        sum+= res[i];
    out = sum;
#ifdef MPI_USED
    MPI_Reduce(&sum, &out, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
    if (myID == 0) 
       cout<<"1st control sum "<< out <<endl;

    sum = 0;
    for (int i = 0; i < EE->N; i++)
        sum+=vec[i];
    out = sum;
#ifdef MPI_USED
    MPI_Reduce(&sum, &out, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif
    if (myID == 0) 
       cout<<"2nd control sum "<< out <<endl;
}

void bestRatio(int Nx, int Ny, int P, int& px, int& py)
{
  vector<int> divisors;
  for (int i=1; i<P+1; i++)
    if (P%i == 0 && i<=Nx && P/i<=Ny)
	    divisors.push_back(i);
  long int min = Nx*Ny;
  for (std::vector<int>::iterator p = divisors.begin(); p != divisors.end(); ++p) {
    //biggest subregion square minimization
    long int tmp = (Nx%(*p)>0?Nx/(*p)+1:Nx/(*p)) * (Ny*(*p)%P>0?Ny*(*p)/P+1:Ny*(*p)/P);
    if (tmp < min || tmp == min && abs((*p)-P/(*p)) < abs(px-py)) {
       min = tmp;
       px = (*p);
       py = P/(*p);
    }
  }
}


int main(int argc, char**argv) {
  int myID = 0; // process ID
  int NumProc = 1; // number of processes 
#ifdef MPI_USED
  int mpi_res;
  mpi_res = MPI_Init(&argc, &argv);
  mpi_res = MPI_Comm_rank(MPI_COMM_WORLD,&myID);
  mpi_res = MPI_Comm_size(MPI_COMM_WORLD,&NumProc);
#endif
  if (argc<5) {
      if (myID == 0)
#ifdef OMP_USED
	  cout<<"Incorrect command line arguments!"<<endl
		  <<"The program must be run with 5 integers passed: Nx, Ny, K3, K4, OMP_NUM_THREADS"<<endl;
#else
	  cout<<"Incorrect command line arguments!"<<endl
		  <<"The program must be run with 4 integers passed: Nx, Ny, K3, K4"<<endl;
#endif
      return -1;
  }
  int Nx = atoi(argv[1]), Ny = atoi(argv[2]), K3 = atoi(argv[3]), K4 = atoi(argv[4]);
#ifdef OMP_USED
  int num_threads = 1;
  if (argc > 5)
     num_threads = atoi(argv[5]);
#endif

  if (Nx < 1 || Ny < 1 || K3 < 1 || K4 < 1 || NumProc > Nx*Ny
#ifdef OMP_USED
		                           || num_threads < 1
#endif
					                     ) {
      if (myID == 0)
	printf("The arguments must be positive integers! Number of processes must not be greater than Nx*Ny\n");
      return -1;
  }
#ifdef OMP_USED
  omp_set_num_threads(num_threads);
#endif
  Mesh mesh;
  int px = 1,py = NumProc;
  bestRatio(Nx, Ny, NumProc, px, py);
  mesh.generate(Nx,Ny,K3,K4,px,py);
  
  //struct rusage usage;
  //getrusage(RUSAGE_SELF, &usage);
  struct timeval timer;
  //timer = usage.ru_utime;
  gettimeofday(&timer, NULL);
  double time = timer.tv_sec*1e6 + timer.tv_usec;
#ifdef MPI_USED
  if (NumProc > 1)
    mesh.refreshHalo();
#endif
  
  mesh.SpMV();
#ifdef MPI_USED
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  //getrusage(RUSAGE_SELF, &usage);
  //timer = usage.ru_utime;
  gettimeofday(&timer, NULL);
  time = timer.tv_sec*1e6 + timer.tv_usec - time;
  if (myID == 0) {
     cout<<"px = "<<px <<", py = "<<py<<endl;
     cout << "time taken   "<<time/1000<<"  milliseconds"<<endl;
  }
  mesh.printControlSums();
  //mesh.print(NumProc);
  mesh.printRes(Nx,Ny,K3,K4);
#ifdef MPI_USED
  MPI_Finalize();
#endif
  return 0;
}
