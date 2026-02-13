#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> // Per strcmp e per gestire le stringhe
#include "func.h"
#include "helpers.h"


int main(int argc, char *argv[]) {
    double fixed_target_dist = 0;  // Default: target on origin


    if (argc != 2) {
        fprintf(stderr, "Error! Add a configuration file.\n");
        return EXIT_FAILURE;
    }

    const char* conf_path = argv[1];

    // load configuration
    Config cfg = {0};
    load_config(conf_path, &cfg);

    printf("## Starting Levy Search Simulations on 3D Torus\n");

    char *output_filename = build_output_path(cfg.save_directory, cfg.file_name);
    
    // create or overwrite output file


    FILE *output_file = fopen(output_filename, "w");
    if (output_file == NULL) {
        perror("Error opening output file");
        return EXIT_FAILURE;
    }
    fprintf(output_file, "n_walkers,n_volume,mu,lmax,D,surface,TargetShape,n_targets,fixed_target_dist,detection_time,probability,surface_selector,first_touch_steps,second_touch_steps,delta_selector,delta\n");

    long start_total_time = time(NULL);

    long total_inner_iterations = (long)cfg.len_range_side * cfg.len_list_shapes * cfg.len_range_diam *
                                   cfg.len_range_ntargets * cfg.len_range_nwalkers * cfg.len_rangemu_LevyDistrib * cfg.len_range_probability * cfg.len_range_delta;

    // Ora si usa cfg.num_runs preso da riga di comando o il default
    for (int current_trial = 0; current_trial < cfg.num_runs; ++current_trial) {
        printf("\n--- Overall Trial: %d/%d ---\n", current_trial + 1, cfg.num_runs);
        long elapsed_seconds = time(NULL) - start_total_time;
        printf("    %d trials completed, total elapsed time: %.2f minutes\n", current_trial + 1, (double)elapsed_seconds / 60.0);
        if (current_trial > 0) {
            double estimated_remaining_time = ((double)elapsed_seconds / (current_trial + 1)) * (cfg.num_runs - (current_trial + 1));
            printf("    Estimated time remaining: %.2f minutes\n", estimated_remaining_time / 60.0);
        }

        long pbar_counter = 0;

        for (int i_side = 0; i_side < cfg.len_range_side; ++i_side) {
            int side = cfg.range_side[i_side];
            double n_volume = pow(side, 3);
            int lmax_current = side / 2;
            for (int i_mu = 0; i_mu < cfg.len_rangemu_LevyDistrib; ++i_mu) {
                double mu = cfg.rangemu_LevyDistrib[i_mu];  
                double normalization_constant = get_normalization_constant(mu, lmax_current);         
                for (int i_shape = 0; i_shape < cfg.len_list_shapes; ++i_shape) {
                    const char* TargetShape = cfg.list_shapes[i_shape];
                    for (int i_D = 0; i_D < cfg.len_range_diam; ++i_D) {
                        double D; 
                        if (cfg.surface_selector) {
                            D = get_diameter_from_surface(cfg.range_diam[i_D], TargetShape);
                            //printf("Using surface area %.2f for shape %s gives effective diameter D=%.2f\n", cfg.range_diam[i_D], TargetShape, D);
                        } else if (cfg.delta_selector) {
                            D = cfg.range_diam[i_D]; // in this case, range_diam contains the surface values
                        } else {
                            D = cfg.range_diam[i_D];
                        }
                        double surface = cfg.range_diam[i_D];
                        
                        for (int i_ntargets = 0; i_ntargets < cfg.len_range_ntargets; ++i_ntargets) {
                            int n_targets = cfg.range_ntargets[i_ntargets];
                            for (int i_nwalkers = 0; i_nwalkers < cfg.len_range_nwalkers; ++i_nwalkers) {
                                int n_walkers = cfg.range_nwalkers[i_nwalkers];
                                for (int i_probability = 0; i_probability < cfg.len_range_probability; ++i_probability) {
                                    double probability = cfg.range_probability[i_probability];
                                    for (int i_delta = 0; i_delta < cfg.len_range_delta; ++i_delta) {
                                        double delta = cfg.range_delta[i_delta];
                                        //printf("Running configuration: n_walkers=%d, n_volume=%.0f, mu=%.1f, lmax=%d, D=%.2f, surface=%.2f, TargetShape=%s, n_targets=%d, fixed_target_dist=%.1f, probability=%.2f, delta=%.2f\n",
                                        //        n_walkers, n_volume, mu, lmax_current, D, surface, TargetShape, n_targets, fixed_target_dist, probability, delta);
                                        if (D >= 1 && D <= side ){
                                            Result result = LevySearch3D_MultiWalker(n_walkers, "nest", n_volume, mu, lmax_current,
                                                                                                D, TargetShape, n_targets, fixed_target_dist, probability, normalization_constant, cfg.steps_between, cfg.max_touches, cfg.delta_selector, delta);
                                            double detection_time = result.detection_time;
                                            int first_touch_steps = result.first_touch_steps;
                                            int second_touch_steps = result.second_touch_steps;
                                            fprintf(output_file, "%d,%.0f,%.1f,%d,%.2f,%.2f,%s,%d,%.1f,%.2f,%.2f,%d,%d,%d,%d,%.2f\n",
                                                    n_walkers, n_volume, mu, lmax_current, D, surface, TargetShape, n_targets, fixed_target_dist, detection_time, probability, cfg.surface_selector, first_touch_steps, second_touch_steps, cfg.delta_selector, delta);
                                            }
                                        else {
                                            //printf("Skipping invalid configuration: D=%.2f, side=%d", D, side);
                                        }
                                        pbar_counter++;
                                        if (total_inner_iterations > 0 && pbar_counter % (total_inner_iterations / 100 + 1) == 0) {
                                            printf("\rTrial %d/%d Progress: %.2f%%", current_trial + 1, cfg.num_runs,
                                                (double)pbar_counter * 100.0 / total_inner_iterations);
                                            fflush(stdout);
                                        }
                                
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        printf("\rTrial %d/%d Progress: 100.00%%\n", current_trial + 1, cfg.num_runs);
    }

    long end_total_time = time(NULL);
    printf("\nTotal simulation time: %.2f minutes\n", (double)(end_total_time - start_total_time) / 60.0);

    fclose(output_file);
    printf("Saving results to %s\n", output_filename);
    printf("Save complete.\n");

    free_config(&cfg);            // clean up dynamically allocated memory

    return 0;
}