function [ Means, Stds] = feature_extractor(folder_path, atlas_file, atlas_txt, output_csv_prefix)
    % FEATURE_EXTRACTOR Extracts mean and standard deviation (std) for each ROI from NIfTI images.
    %
    % Description:
    %   This function extracts the mean and standard deviation of voxel intensities
    %   for each region of interest (ROI) in the brain as defined in an atlas.
    %   The image files should be in NIfTI format (.nii/.nii.gz) and should be located
    %   in the specified folder. The function uses an atlas file (.nii) to segment the
    %   brain into regions and then computes the desired statistics for each ROI.
    %
    % Inputs:
    %   - folder_path : string. Path to the folder containing NIfTI image files.
    %   - atlas_file  : string. Path to the atlas NIfTI file used for segmentation.
    %   - atlas_txt   : string. Path to a text file with ROI IDs and names (format: ID<TAB>Name).
    %   - output_csv_prefix (optional) : string. Prefix for the output CSV files.
    %                                      Example: 'results_AD'.
    %
    % Outputs:
    %   - A CSV file for each ROI, containing the mean and standard deviation values.
    %
    % Example:
    %   feature_extractor('path/to/folder', 'path/to/atlas.nii', 'path/to/atlas_labels.txt', 'results_prefix');
    %
    % This function will:
    %   - Load the images and atlas.
    %   - Extract voxel intensities for each ROI.
    %   - Compute the mean and std for each ROI across all images.
    %   - Save results as CSV files if requested.
    
    % Intensity threshold to ignore very small voxel values (e.g., background)
    INTENSITY_THRESHOLD = 1e-6;

    % === 1. Load images from the folder ===
    nifti_files = dir(fullfile(folder_path, '*.nii*'));
    image_filepaths = string(fullfile({nifti_files.folder}, {nifti_files.name}));
    num_images = numel(image_filepaths);
    
    % Check if any NIfTI files were found
    if num_images == 0
        error('No NIfTI files found in the folder: %s', folder_path);
    end

    % === 2. Load the atlas ===
    atlas_data = double(niftiread(atlas_file));
    atlas_size = size(atlas_data);

    % === 3. Load ROI data ===
    fid = fopen(atlas_txt, 'r');
    roi_data = textscan(fid, '%d%s', 'Delimiter', '\t');
    fclose(fid);
    roi_ids = roi_data{1};   % ROI IDs (numeric)
    roi_names = roi_data{2}; % ROI names (strings)
    num_rois = numel(roi_ids);

    % === 4. Pre-calculate ROI masks ===
    % Create logical masks for each ROI, which will later be used to extract voxel values
    roi_masks = false([atlas_size, num_rois]);
    for j = 1:num_rois
        roi_masks(:,:,:,j) = (atlas_data == roi_ids(j));
    end

    % === 5. Pre-allocate results ===
    % Initialize matrices to store the mean and standard deviation values for each image and ROI
    Means = NaN(num_images, num_rois);
    Stds = NaN(num_images, num_rois);

    % === 6. Feature extraction ===
    % Loop through all images and calculate the mean and std for each ROI
    for i = 1:num_images
        % Load the current image as a double precision array
        img = double(niftiread(image_filepaths(i)));

        % Loop through all ROIs and compute the mean and std for each ROI
        for j = 1:num_rois
            % Extract the voxel intensities corresponding to the current ROI
            voxels = img(roi_masks(:,:,:,j));
            
            % Remove very small values (background) by applying the intensity threshold
            voxels = voxels(voxels > INTENSITY_THRESHOLD);
            
            % If there are valid voxels, calculate the mean and std
            if ~isempty(voxels)
                Means(i,j) = mean(voxels);
                Stds(i,j) = std(voxels);
            end
        end
    end

    % === 7. Create separate tables for means and standard deviations ===
    % Prepare table column names for the means and stds for each ROI
    img_names = "AD-" + (1:num_images)';
    mean_colnames = strcat("Mean_", roi_names');
    std_colnames = strcat("Std_", roi_names');

    % Create tables for the means and standard deviations
    MeanTable = array2table(Means, 'VariableNames', mean_colnames);
    MeanTable = addvars(MeanTable, img_names, 'Before', 1, 'NewVariableNames', 'Image');

    StdTable = array2table(Stds, 'VariableNames', std_colnames);
    StdTable = addvars(StdTable, img_names, 'Before', 1, 'NewVariableNames', 'Image');

    % === 8. Display summary ===
    % Print out the tables to the MATLAB command window
    disp('--- Mean Table ---');
    disp(MeanTable);
    disp('--- Standard Deviation Table ---');
    disp(StdTable);

    % === 9. Save to CSV ===
    % If an output prefix is provided, save the tables as CSV files
    if nargin == 4 && ~isempty(output_csv_prefix)
        mean_file = strcat(output_csv_prefix, '_mean.csv');
        std_file = strcat(output_csv_prefix, '_std.csv');
        
        % Write the tables to CSV files
        writetable(MeanTable, mean_file);
        writetable(StdTable, std_file);
        
        % Print the file names to confirm
        fprintf('Tables saved:\n  - %s\n  - %s\n', mean_file, std_file);
    end
end