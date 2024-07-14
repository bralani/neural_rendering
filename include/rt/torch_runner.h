#ifndef RT_TORCH_RUNNER_H
#define RT_TORCH_RUNNER_H
void set_model_path(const char* model_path);
#ifdef __cplusplus
extern "C" {
#endif

	int run_torch(double* para);

#ifdef __cplusplus
}
#endif
typedef enum {
    normal = 1,
    neural = 2
} render_type;
#endif // !RT_TORCH_RUNNER_H