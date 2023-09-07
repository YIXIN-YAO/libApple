#include "libApple.h"
#include <iostream>
#include <csignal>
using namespace std;
static void sig(int signum) {
    runtime_stop();
}
int main() {
    const char* device = "044422250475";
    bool run = runtime_start(nullptr,device);
    if(!run){
        return -1;
    }
    signal(SIGINT, sig);
    while (true)
    {
        std::printf("distance %f\n",detect_tree_distance(device));
    }

    return 0;
}