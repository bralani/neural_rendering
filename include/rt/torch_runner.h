#ifndef RT_TORCH_RUNNER_H
#define RT_TORCH_RUNNER_H

#ifdef __cplusplus
extern "C" {
#endif

void set_model_path(const char* model_path);

int run_torch(double* para);

#ifdef __cplusplus
}
#endif

typedef enum {
    normal = 1,
    neural = 2
} render_type;

#endif /* RT_TORCH_RUNNER_H */


/*
 * Local Variables:
 * tab-width: 8
 * mode: C
 * indent-tabs-mode: t
 * c-file-style: "stroustrup"
 * End:
 * ex: shiftwidth=4 tabstop=8
 */
