/*
 * this program is to verify the correctness of 
 * floating-free perceptron's behaviour
 * modify simulate_function to a function
 * that is simulated for proof
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "pow2.h"
#include "sigmoid.h"

#define L 2
#define M 7
#define N 1
#define ETA 3
#define ALPHA 16
#define BETA 16
#define GAMMA 16
#define DELTA 16
#define LOOP_MAX 10000
#define HIS_LEN 128
#define ll long long
#define RANGE 100

/*parceptron parameters*/
struct perceptron_param{
    ll wlm[L+1][M];
    ll wmn[M+1][N];
    ll dlm[L+1][M];
    ll dmn[M+1][N];
    ll Lout[L];
    ll Min[M];
    ll Mout[M];
    ll Nin[N];
}p_param;

struct teacher{
    int x[HIS_LEN];
    int y[HIS_LEN];
    int ans[HIS_LEN];
}t;

ll simulate_function(int x){
    return abs(20-x)+20;
//return 40*sin(x/10)+40;
}

static void initialize_perceptron(void){
    int i,j;
    for(i=0;i<L+1;i++){
        for(j=0;j<M;j++){
            p_param.wlm[i][j] = rand() % (pow2[DELTA+1]+1) - pow2[DELTA];
        }
    }
    for(i=0;i<M+1;i++){
        for(j=0;j<N;j++){
            p_param.wmn[i][j] = rand() % (pow2[DELTA+1]+1) - pow2[DELTA];
        }
    }
}

static void initialize_edge_delta(void){
    int i,j;
    for(i=0;i<L+1;i++){
        for(j=0;j<M;j++){
            p_param.dlm[i][j] = 0;
        }
    }
    for(i=0;i<M+1;i++){
        for(j=0;j<N;j++){
            p_param.dmn[i][j] = 0;
        }
    }
}

static ll get_prediction(int x, int y){
    ll modin;
    int i,j;
    p_param.Lout[0] = x;
    p_param.Lout[1] = y;

    //M層のi-thノードに対する入力値を計算する
    for(i=0;i<M;i++){
        p_param.Min[i] = 0;
        //Lout * weightの和を計算
        for(j=0;j<L;j++){
            p_param.Min[i] += p_param.wlm[j][i] * p_param.Lout[j];
        }
        //M層のi番目ノードの閾値分を入力から減算
        p_param.Min[i] += p_param.wlm[L][i] * -1;
    }

    //M層i-thノードのoutputを計算する    
    for(i=0;i<M;i++){
        modin = (p_param.Min[i] >> (1 + DELTA - ALPHA)) / BETA + pow2[ALPHA-1];
        if(0 <= modin && modin < (1 << ALPHA)){
            p_param.Mout[i] = sigmoid[modin];
        }else if(modin < 0){
            p_param.Mout[i] = 0;
        }else{
            p_param.Mout[i] = 1 << GAMMA;
        }
    }

    //N層i-thノードへの入力値を計算する
    for(i=0;i<N;i++){
        p_param.Nin[i] = 0;
        for(j=0;j<M;j++){
            //M層output * weightの和を計算
            p_param.Nin[i] += p_param.wmn[j][i] * p_param.Mout[j];
        }
        p_param.Nin[i] += p_param.wmn[M][i] * -1;
    }

    modin = (p_param.Nin[0] >> (1+GAMMA+DELTA-ALPHA)) / BETA + pow2[ALPHA-1];
    if(0 <= modin && modin < (1 << ALPHA)){
        return sigmoid[modin];
    }else if(modin < 0){
        return 0;
    }else{
        return 1 << GAMMA;
    }
}

static void train(){
    ll result, delta_k, delta_j;
    int x,i,j,k;
    int ans;

    initialize_perceptron();

    for(x=0;x<LOOP_MAX;x++){
        //差分変数の初期化
        initialize_edge_delta();

        //全ての教師データに対して
        for(i=0;i<HIS_LEN;i++){
            //教師データを取得する必要がある
            ans = t.ans[i];

            //予測を出す
            result = get_prediction(t.x[i],
                                    t.y[i]);

            delta_k = (ans << GAMMA) - result;
            delta_k *= (1 << GAMMA) - result;
            delta_k >>= GAMMA;
            delta_k *= result;
            delta_k >>= GAMMA;

            //M->Nの偏微分値
            for(j=0;j<M+1;j++){
                for(k=0;k<N;k++){
                    if(j != M){
                        p_param.dmn[j][k] += (((delta_k * p_param.Mout[j]) >> GAMMA) << DELTA) >> GAMMA;
                    }else{
                        p_param.dmn[j][k] += ((delta_k * -1) << DELTA) >> GAMMA;
                    }
                }
            }

            //L->Mの偏微分値
            for(j=0;j<M;j++){
                delta_j = (delta_k * p_param.wmn[j][0]) >> DELTA;
                delta_j *= p_param.Mout[j];
                delta_j >>= GAMMA;
                delta_j *= (1<<GAMMA) - p_param.Mout[j];
                delta_j >>= GAMMA;
                for(k=0;k<L+1;k++){
                    if(k != L){
                        p_param.dlm[k][j] += (((delta_j * p_param.Lout[k]) >> GAMMA) << DELTA) >> GAMMA;
                    }else{
                        p_param.dlm[k][j] += ((delta_j * -1) << DELTA) >> GAMMA;
                    }
                }
            }
        }
        for(i=0;i<L+1;i++){
            for(j=0;j<M;j++){
                p_param.wlm[i][j] += p_param.dlm[i][j] >> ETA;
            }
        }
        for(i=0;i<M+1;i++){
            for(j=0;j<N;j++){
                p_param.wmn[i][j] += p_param.dmn[i][j] >> ETA;
            }
        }
    }
}

int main(void){
    int i;
    int lb,ub,mid;

    //generate teacher data
    srand(time(NULL));
    for(i=0;i<HIS_LEN;i++){
        t.x[i] = rand() % (RANGE+1);
        t.y[i] = rand() % (RANGE+1);
        if(t.y[i] <= simulate_function(t.x[i])){
            t.ans[i] = 0;
        }else{
            t.ans[i] = 1;
        }
        printf("%d %d %d\n", t.x[i], t.y[i], t.ans[i]);
    }

    printf("\n\n");

    //train perceptron
    train();

    //output changing value using binary search
    for(i=1;i<RANGE+1;i++){
        lb = 0;
        ub = 1 << 20;
        while(ub-lb>1){
            mid = (ub+lb)/2;
            if(get_prediction(i,mid) < (1 << (GAMMA - 1))){
                lb = mid;
            }else{
                ub = mid;
            }
        }
        printf("%d %d\n", i, lb);
    }

    printf("\n\n");
    for(i=1;i<RANGE+1;i++){
        printf("%d %d\n", i, simulate_function(i));
    }
    return 0;
}
