#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> // Per strcmp e per gestire le stringhe
#include "func.h"
#include "helpers.h"

double get_diameter_from_measure_and_shape(double measure, const char* shape) {
    if (measure <= 0) {
        fprintf(stderr, "Error: The measure (volume/area/length) must be positive.\n");
        return -1.0;
    }

    if (strcmp(shape, "Ball") == 0) {
        // Measure is 3D Volume for a Sphere
        // V = (1/6) * PI * D^3
        // D = cbrt(6 * V / PI)
        return cbrt(6.0 * measure / M_PI);
    } else if (strcmp(shape, "Disk") == 0) {
        // Measure is 2D Area for a Circle (Disk)
        // A = (1/4) * PI * D^2
        // D = sqrt(4 * A / PI)
        return sqrt(4.0 * measure / M_PI);
    } else if (strcmp(shape, "Line") == 0) {
        // Measure is 1D Length for a Line
        // L = D
        // V = pi * D + 4/3 pi
        return measure;
    } else {
        fprintf(stderr, "Error: Unknown shape '%s'. Supported shapes are 'Ball', 'Disk', 'Line'.\n", shape);
        return -1.0; // Indicate error
    }
}

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
    fprintf(output_file, "n_walkers,n_volume,mu,lmax,D,TargetShape,DiskDiameter,n_targets,fixed_target_dist,detection_time,probability\n");

    long start_total_time = time(NULL);

    long total_inner_iterations = (long)cfg.len_range_side * cfg.len_list_shapes * cfg.len_range_disk_diameter *
                                   cfg.len_range_ntargets * cfg.len_range_nwalkers * cfg.len_rangemu_LevyDistrib * cfg.len_range_probability;
    
    printf("Warning: this code considers the volume of the extended target (target + boundary) for the calculation of D.\n");

    // Ora si usacfg.num_runs preso da riga di comando o il default
    for (int current_trial = 0; current_trial < cfg.num_runs; ++current_trial) {
        printf("\n--- Overall Trial: %d/%d ---\n", current_trial + 1, cfg.num_runs);
        long elapsed_seconds = time(NULL) - start_total_time;
        printf("    %d trials completed, total elapsed time: %.2f minutes\n", current_trial + 1, (double)elapsed_seconds / 60.0);
        if (current_trial > 0) {
            double estimated_remaining_time = ((double)elapsed_seconds / (current_trial + 1)) * (cfg.num_runs - (current_trial + 1));
            printf("    Estimated time remaining: %.2f minutes\n", estimated_remaining_time / 60.0);
        }

        long pbar_counter = 0;
        int temp1;
        double temp2;

        for (int i_side = 0; i_side < cfg.len_range_side; ++i_side) {

            int side = cfg.range_side[i_side];
            double n_volume = pow(side, 3);
            int lmax_current = side / 2;
            for (int i_mu = 0; i_mu < cfg.len_rangemu_LevyDistrib; ++i_mu) {

                double mu = cfg.rangemu_LevyDistrib[i_mu];
                // compute normalization constant
                double a = get_normalization_constant(mu, lmax_current);
                for (int i_disk_diameter = 0; i_disk_diameter < cfg.len_range_disk_diameter; ++i_disk_diameter) {
                    double disk_diameter = cfg.range_disk_diameter[i_disk_diameter];
                    for (int i_ntargets = 0; i_ntargets < cfg.len_range_ntargets; ++i_ntargets) {
                        int n_targets = cfg.range_ntargets[i_ntargets];
                        for (int i_nwalkers = 0; i_nwalkers < cfg.len_range_nwalkers; ++i_nwalkers) {
                            int n_walkers = cfg.range_nwalkers[i_nwalkers];
                            for (int i_shape = 0; i_shape < cfg.len_list_shapes; ++i_shape) {

                                const char* TargetShape = cfg.list_shapes[i_shape];
                                double D;
                                if (strcmp(TargetShape, cfg.reference_shape) == 0) {
                                    D = disk_diameter;
                                } else if (strcmp(TargetShape, "Ball_no_boundary") == 0) {
                                    //double target_volume = pow(disk_diameter+2, 2) * M_PI / 4.0 * 2; // volume of the extended disk
                                    double target_volume = 4*M_PI + 2 * disk_diameter * M_PI; // superface of the extended line 
                                    D = get_diameter_from_measure_and_shape(target_volume, "Ball");
                                } else {
                                    fprintf(stderr, "Error: Unsupported shape '%s'.\n", TargetShape);
                                    continue; // Skip unsupported shapes
                                }
                                for (int i_prob = 0; i_prob < cfg.len_range_probability; ++i_prob) {
                                    double p = cfg.range_probability[i_prob];
                                    //printf("Running simulation with n_walkers=%d, n_volume=%.0f, mu=%.1f, lmax=%d, D=%.2f, TargetShape=%s, DiskDiameter=%.1f, n_targets=%d, fixed_target_dist=%.1f, p=%.2f\n",
                                    //       n_walkers, n_volume, mu, lmax_current, D, TargetShape, disk_diameter, n_targets, fixed_target_dist, p);
                                    Result result = LevySearch3D_MultiWalker(n_walkers, "nest", n_volume, mu, lmax_current,
                                                                                 D, TargetShape, n_targets, fixed_target_dist, p, a, 2, 0, 0, 0.0);
                                    double detection_time = result.detection_time;
                                    int first_touch_steps = result.first_touch_steps;
                                    int second_touch_steps = result.second_touch_steps;  
                                    // double detection_time = LevySearch3D_MultiWalker_jump_and_random_dir(n_walkers, "nest", n_volume, mu, lmax_current, D, TargetShape, n_targets, fixed_target_dist, 1.0, &temp1, &temp2);
                                    //printf("Detection time: %.2f\n", detection_time);
                                    fprintf(output_file, "%d,%.0f,%.1f,%d,%.2f,%s,%.0f,%d,%.1f,%.2f,%.2f\n",
                                            n_walkers, n_volume, mu, lmax_current, D, TargetShape, disk_diameter, n_targets, fixed_target_dist, detection_time, p, 0, 2);
                                    
                                    pbar_counter++;
                                    if (total_inner_iterations > 0 && pbar_counter % (total_inner_iterations / 100 + 1) == 0) {
                                        printf("\rTrial %d/%d Progress: %.2f%%", current_trial + 1,cfg.num_runs,
                                            (double)pbar_counter * 100.0 / total_inner_iterations);
                                        fflush(stdout);
                                    }
                            }   }
                        }
                    }
                }
            }
        }
        printf("\rTrial %d/%d Progress: 100.00%%\n", current_trial + 1,cfg.num_runs);
    }

    long end_total_time = time(NULL);
    printf("\nTotal simulation time: %.2f minutes\n", (double)(end_total_time - start_total_time) / 60.0);
    free_config(&cfg); 
    fclose(output_file);
    printf("Saving results to %s\n", output_filename);
    printf("Save complete.\n");

    return 0;
}
